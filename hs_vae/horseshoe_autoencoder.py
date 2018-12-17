""" Uses a non-centered parameterization of the model.
    Fully factorized Gaussian + IGamma Variational distribution
	q = N(w_ijl | m_ijl, sigma^2_ijl) N(ln \tau_kl | params) IGamma(\lambda_kl| params)
	IGamma(\tau_l | params) IGamma(\lambda_l| params)
"""

import autograd.numpy.random as npr
import autograd.numpy as ag_np
from data.utils import *
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln, psi
from src.utility_functions import diag_gaussian_entropy, inv_gamma_entropy, log_normal_entropy
from autograd import grad
from hs_vae.base_autoencoder import VAE
from src.optimizers import *
from copy import copy



class NeuralNetworkAutoGrad():
    def __init__(self, nn_structure=[32], n_dims_input=1, n_dims_output=1, \
                 weight_fill_func=np.zeros, bias_fill_func=np.zeros, activation_func=lambda x: ag_np.maximum(0, x)):

        self.nn_param_list = []
        self.activation_func = activation_func
        self.n_dims_input = n_dims_input
        self.n_dims_output = n_dims_output

        n_hiddens_per_layer_list = [n_dims_input] + nn_structure + [n_dims_output]

        # Given full network size list is [a, b, c, d, e]
        # For loop should loop over (a,b) , (b,c) , (c,d) , (d,e)
        for n_in, n_out in zip(n_hiddens_per_layer_list[:-1], n_hiddens_per_layer_list[1:]):
            self.nn_param_list.append(
                dict(
                    w=weight_fill_func((n_in, n_out)),
                    b=bias_fill_func((n_out,)),
                ))

    def forward(self, x):
        for layer_id, layer_dict in enumerate(self.nn_param_list):
            if layer_id == 0:
                if x.ndim > 1:
                    in_arr = x
                else:
                    if x.size == self.nn_param_list[0]['w'].shape[0]:
                        in_arr = x[ag_np.newaxis, :]
                    else:
                        in_arr = x[:, ag_np.newaxis]
            else:
                in_arr = self.activation_func(out_arr)
            out_arr = ag_np.dot(in_arr, layer_dict['w']) + layer_dict['b']
        return ag_np.squeeze(out_arr)


class FactorizedHierarchicalInvGamma:
    def __init__(self, n_weights, lambda_a, lambda_b, lambda_b_global, tau_a, shapes, train_stats, classification=True,
                 n_data=None):
        self.name = "Factorized Hierarchical Inverse Gamma Variational Approximation"
        self.classification = classification
        self.n_weights = n_weights
        self.shapes = shapes
        self.num_hidden_layers = len(shapes) - 1
        self.lambda_a_prior = lambda_a
        self.lambda_b_prior = lambda_b
        self.lambda_a_prior_global = 0.5
        self.lambda_b_prior_global = lambda_b_global
        self.lambda_a_prior_oplayer = 0.5
        self.lambda_b_prior_oplayer = 1.
        self.tau_a_prior = tau_a
        self.tau_a_prior_global = 0.5
        self.tau_a_prior_oplayer = 0.5
        self.l2pi = np.log(2 * np.pi)
        self.n_data = n_data
        self.noise_entropy = None

    ######### PACK UNPACK PARAMS #################################################
    def initialize_variational_params(self, param_scale=1):
        # Initialize weights
        wlist = list()
        for m, n in self.shapes:
            wlist.append(npr.randn(m * n) * np.sqrt(2 / m))
            wlist.append(np.zeros(n))  # bias
        w = np.concatenate(wlist)
        log_sigma = param_scale * npr.randn(w.shape[0]) - 10.
        # initialize scale parameters
        self.tot_outputs = 0
        for _, num_hl_outputs in self.shapes:
            self.tot_outputs += num_hl_outputs
        # No hs priors on the outputs
        self.tot_outputs = self.tot_outputs - self.shapes[-1][1]

        tau_mu, tau_log_sigma, tau_global_mu, tau_global_log_sigma, tau_oplayer_mu, tau_oplayer_log_sigma = \
            self.initialize_scale_from_prior()
        init_params = np.concatenate([w.ravel(), log_sigma.ravel(),
                                      tau_mu.ravel(), tau_log_sigma.ravel(), tau_global_mu.ravel(),
                                      tau_global_log_sigma.ravel(), tau_oplayer_mu, tau_oplayer_log_sigma])

        return init_params

    def initialize_scale_from_prior(self):
        # scale parameters (hidden + observed),
        self.lambda_a_hat = (self.tau_a_prior + self.lambda_a_prior) * np.ones([self.tot_outputs, 1]).ravel()
        self.lambda_b_hat = (1.0 / self.lambda_b_prior ** 2) * np.ones([self.tot_outputs, 1]).ravel()
        self.lambda_a_hat_global = (self.tau_a_prior_global + self.lambda_a_prior_global) \
                                   * np.ones([self.num_hidden_layers, 1]).ravel()
        self.lambda_b_hat_global = (1.0 / self.lambda_b_prior_global ** 2) * np.ones(
            [self.num_hidden_layers, 1]).ravel()
        # set oplayer lambda param
        self.lambda_a_hat_oplayer = np.array(self.tau_a_prior_oplayer + self.lambda_a_prior_oplayer).reshape(-1)
        self.lambda_b_hat_oplayer = (1.0 / self.lambda_b_prior_oplayer ** 2) * np.ones([1]).ravel()
        # sample from half cauchy and log to initialize the mean of the log normal
        sample = np.abs(self.lambda_b_prior * (npr.randn(self.tot_outputs) / npr.randn(self.tot_outputs)))
        tau_mu = np.log(sample)
        tau_log_sigma = npr.randn(self.tot_outputs) - 10.
        # one tau_global for each hidden layer
        sample = np.abs(
            self.lambda_b_prior_global * (npr.randn(self.num_hidden_layers) / npr.randn(self.num_hidden_layers)))
        tau_global_mu = np.log(sample)
        tau_global_log_sigma = npr.randn(self.num_hidden_layers) - 10.
        # one tau for all op layer weights
        sample = np.abs(self.lambda_b_hat_oplayer * (npr.randn() / npr.randn()))
        tau_oplayer_mu = np.log(sample)
        tau_oplayer_log_sigma = npr.randn(1) - 10.

        return tau_mu, tau_log_sigma, tau_global_mu, tau_global_log_sigma, tau_oplayer_mu, tau_oplayer_log_sigma

    def unpack_params(self, params):
        # unpack params
        w_vect = params[:self.n_weights]
        num_std = 2 * self.n_weights

        sigma = np.log(1 + np.exp(params[self.n_weights:num_std]))

        tau_mu = params[num_std:num_std + self.tot_outputs]
        tau_sigma = np.log(
            1 + np.exp(params[num_std + self.tot_outputs:num_std + 2 * self.tot_outputs]))
        tau_mu_global = params[num_std + 2 * self.tot_outputs: num_std + 2 * self.tot_outputs + self.num_hidden_layers]
        tau_sigma_global = np.log(1 + np.exp(params[num_std + 2 * self.tot_outputs + self.num_hidden_layers:num_std +
                                                                                                            2 * self.tot_outputs + 2 * self.num_hidden_layers]))
        tau_mu_oplayer = params[num_std + 2 * self.tot_outputs + 2 * self.num_hidden_layers: num_std +
                                                                                             2 * self.tot_outputs + 2 * self.num_hidden_layers + 1]
        tau_sigma_oplayer = np.log(
            1 + np.exp(params[num_std + 2 * self.tot_outputs + 2 * self.num_hidden_layers + 1:]))

        return w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer

    def unpack_layer_weight_variances(self, sigma_vect):
        for m, n in self.shapes:
            yield sigma_vect[:m * n].reshape((m, n)), sigma_vect[m * n:m * n + n]
            sigma_vect = sigma_vect[(m + 1) * n:]

    def unpack_layer_weight_priors(self, tau_vect):
        for m, n in self.shapes:
            yield tau_vect[:n]
            tau_vect = tau_vect[n:]

    def unpack_layer_weights(self, w_vect):
        for m, n in self.shapes:
            yield w_vect[:m * n].reshape((m, n)), w_vect[m * n:m * n + n]
            w_vect = w_vect[(m + 1) * n:]

    ######### Fixed Point Updates ################################## #####
    def fixed_point_updates(self, params):
        if self.classification:
            w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
                self.unpack_params(params)
        else:
            w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, _, _ \
                = self.unpack_params(params)
        # update lambda moments
        self.lambda_b_hat = np.exp(-tau_mu + 0.5 * tau_sigma ** 2) + (1. / self.lambda_b_prior ** 2)
        self.lambda_b_hat_global = np.exp(-tau_mu_global + 0.5 * tau_sigma_global ** 2) + (
            1. / self.lambda_b_prior_global ** 2)
        self.lambda_b_hat_oplayer = np.exp(-tau_mu_oplayer + 0.5 * tau_sigma_oplayer ** 2) + (
            1. / self.lambda_b_prior_oplayer ** 2)
        return None

    ######### ELBO CALC ################################################
    def forward(self, mu_vect, sigma_vect, tau_mu_vect, tau_sigma_vect, tau_mu_global, tau_sigma_global,
                tau_mu_oplayer, tau_sigma_oplayer, inputs):
        for layer_id, (mu, var, tau_mu, tau_sigma) in enumerate(
                zip(self.unpack_layer_weights(mu_vect), self.unpack_layer_weight_variances(sigma_vect),
                    self.unpack_layer_weight_priors(tau_mu_vect),
                    self.unpack_layer_weight_priors(tau_sigma_vect))):
            w, b = mu
            sigma__w, sigma_b = var
            if layer_id < len(self.shapes) - 1:
                scale_mu = 0.5 * (tau_mu + tau_mu_global[layer_id])
                scale_v = 0.25 * (tau_sigma ** 2 + tau_sigma_global[layer_id] ** 2)
                scale = np.exp(scale_mu + np.sqrt(scale_v) * npr.randn(tau_mu.shape[0]))
                mu_w = np.dot(inputs, w) + b
                v_w = np.dot(inputs ** 2, sigma__w ** 2) + sigma_b ** 2
                outputs = (np.sqrt(v_w) / np.sqrt(inputs.shape[1])) * np.random.normal(size=mu_w.shape) + mu_w
                outputs = scale * outputs
                inputs = outputs * (outputs > 0)
            else:
                op_scale_mu = 0.5 * tau_mu_oplayer
                op_scale_v = 0.25 * tau_sigma_oplayer ** 2
                Ekappa_half = np.exp(op_scale_mu + np.sqrt(op_scale_v) * npr.randn())
                mu_w = np.dot(inputs, w) + b
                v_w = np.dot(inputs ** 2, sigma__w ** 2) + sigma_b ** 2
                outputs = Ekappa_half * (np.sqrt(v_w) / np.sqrt(inputs.shape[1])) * np.random.normal(
                    size=mu_w.shape) + mu_w
        return outputs

    def EPw_Gaussian(self, prior_precision, w, sigma):
        """"\int q(z) log p(z) dz, assuming gaussian q(z) and p(z)"""
        wD = w.shape[0]
        prior_wvar_ = 1. / prior_precision
        a = - 0.5 * wD * np.log(2 * np.pi) - 0.5 * wD * np.log(prior_wvar_) - 0.5 * prior_precision * (
            np.dot(w.T, w) + np.sum((sigma ** 2)))
        return a

    def EP_Gamma(self, Egamma, Elog_gamma):
        """ Enoise precision """
        return self.noise_a * np.log(self.noise_b) - gammaln(self.noise_a) + (
                                                                                 - self.noise_a - 1) * Elog_gamma - self.noise_b * Egamma

    def EPtaulambda(self, tau_mu, tau_sigma, tau_a_prior, lambda_a_prior,
                    lambda_b_prior, lambda_a_hat, lambda_b_hat):
        """ E[ln p(\tau | \lambda)] + E[ln p(\lambda)]"""
        etau_given_lambda = -gammaln(tau_a_prior) - tau_a_prior * (np.log(lambda_b_hat) - psi(lambda_a_hat)) + (
                                                                                                                   -tau_a_prior - 1.) * tau_mu - np.exp(
            -tau_mu + 0.5 * tau_sigma ** 2) * (lambda_a_hat /
                                               lambda_b_hat)
        elambda = -gammaln(lambda_a_prior) - 2 * lambda_a_prior * np.log(lambda_b_prior) + (-lambda_a_prior - 1.) * (
            np.log(lambda_b_hat) - psi(lambda_a_hat)) - (1. / lambda_b_prior ** 2) * (lambda_a_hat / lambda_b_hat)
        return np.sum(etau_given_lambda) + np.sum(elambda)

    def entropy(self, sigma, tau_sigma, tau_mu, tau_sigma_global, tau_mu_global, tau_sigma_oplayer, tau_mu_oplayer):
        ent_w = diag_gaussian_entropy(np.log(sigma), self.n_weights)
        ent_tau = log_normal_entropy(np.log(tau_sigma), tau_mu, self.tot_outputs) + log_normal_entropy(
            np.log(tau_sigma_global), tau_mu_global, self.num_hidden_layers) + log_normal_entropy(
            np.log(tau_sigma_oplayer), tau_mu_oplayer, 1)
        ent_lambda = inv_gamma_entropy(self.lambda_a_hat, self.lambda_b_hat) + inv_gamma_entropy(
            self.lambda_a_hat_global, self.lambda_b_hat_global) + inv_gamma_entropy(self.lambda_a_hat_oplayer,
                                                                                    self.lambda_b_hat_oplayer)
        return ent_w, ent_tau, ent_lambda


class HS_VAE(VAE):
    def __init__(
            self,
            q_sigma=0.2,
            n_dims_code=16,
            n_dims_data=64,
            hidden_layer_sizes=[32],
            classification=True,
            batch_size=128,
            lambda_b_global=1.0,
            warm_up=False,
            polyak=False, ):

        super(HS_VAE, self).__init__()
        layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
        )
        self.n_dims_code = n_dims_code
        self.q_sigma = q_sigma
        self.n_dims_data = n_dims_data

        self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.layer_sizes = layer_sizes
        self.lambda_b_global = lambda_b_global
        self.N_weights = sum((m + 1) * n for m, n in self.shapes)

        self.elbo = list()
        self.val_ll = list()
        self.val_err = list()
        self.train_err = list()
        self.test_err = list()

        self.variational_params = None
        self.init_params = None
        self.polyak_params = None
        self.polyak = polyak
        self.variational_params_store = {}
        self.optimal_elbo_params = None
        self.warm_up = warm_up  # if True, anneal in KL

        # TODO: in Ghosh's implementation, mu is the mean of all x_train and
        # TODO: sigma is the standard deviation of all x_train
        train_stats = dict()
        train_stats['mu'] = 0
        train_stats['sigma'] = 1

        # Initialize the encoder and decoder
        # TODO: incorporate Factorized Hierarchical Inverse Gamma
        self.horseshoe_encoder = FactorizedHierarchicalInvGamma(
            lambda_a=0.5, lambda_b=1.0,
            lambda_b_global=self.lambda_b_global, tau_a=0.5,
            shapes=self.shapes, train_stats=train_stats,
            classification=classification,
            n_weights=self.N_weights)

        self.decoder = NeuralNetworkAutoGrad(nn_structure=hidden_layer_sizes, \
                                             n_dims_input=n_dims_code, n_dims_output=n_dims_data)

    def neg_elbo(self, params, epoch, xs_ND, ys_ND, matrix_entries):
        """

        :param params:
        :param epoch:
        :param xs_ND:
        :param ys_ND:
        :param matrix_entries:
        :return:
        """
        if self.warm_up:
            nt = 200  # linear increments between 0 and 1 up to nt (1 after nt)
            temperature = epoch / nt
            if temperature > 1:
                temperature = 1
        else:
            temperature = 1

        # Unpack the current parameters using function provided by Inv-Gamma
        w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer \
            = self.horseshoe_encoder.unpack_params(params)

        # Compute the x_NC codes and y_NC codes (latent embeddings)
        xs_NC = self.horseshoe_encoder.forward(w_vect, sigma, tau_mu, tau_sigma, \
                                               tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer,
                                               xs_ND)

        ys_NC = self.horseshoe_encoder.forward(w_vect, sigma, tau_mu, tau_sigma, \
                                               tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer,
                                               ys_ND)

        # Compute log likelihood
        # TODO: check if there is any scaling factor I need to compute based on implementions in InvGamma
        log_lik = self.log_likelihood_compute(xs_NC, ys_NC, xs_ND, ys_ND, matrix_entries)

        # Compute the entropies and log_prior
        # Note that this is taken from implementations from inv-Gamma
        log_prior, ent_w, ent_tau, ent_lambda = self.entropy_compute(params)

        log_variational = ent_w + ent_tau + ent_lambda
        minibatch_rescaling = 1. / self.M
        ELBO = temperature * minibatch_rescaling * (log_variational + log_prior) + log_lik
        return -1 * ELBO

    def entropy_compute(self, params):
        w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer \
            = self.horseshoe_encoder.unpack_params(params)

        log_prior = self.horseshoe_encoder.EPw_Gaussian(1., w_vect, sigma)
        log_prior = log_prior + \
                    self.horseshoe_encoder.EPtaulambda(tau_mu, tau_sigma, self.horseshoe_encoder.tau_a_prior, \
                                                       self.horseshoe_encoder.lambda_a_prior,
                                                       self.horseshoe_encoder.lambda_b_prior,
                                                       self.horseshoe_encoder.lambda_a_hat,
                                                       self.horseshoe_encoder.lambda_b_hat) + \
                    self.horseshoe_encoder.EPtaulambda(tau_mu_global, tau_sigma_global,
                                                       self.horseshoe_encoder.tau_a_prior_global,
                                                       self.horseshoe_encoder.lambda_a_prior_global,
                                                       self.horseshoe_encoder.lambda_b_prior_global,
                                                       self.horseshoe_encoder.lambda_a_hat_global,
                                                       self.horseshoe_encoder.lambda_b_hat_global) + \
                    self.horseshoe_encoder.EPtaulambda(tau_mu_oplayer, tau_sigma_oplayer,
                                                       self.horseshoe_encoder.tau_a_prior_oplayer,
                                                       self.horseshoe_encoder.lambda_a_prior_oplayer,
                                                       self.horseshoe_encoder.lambda_b_prior_oplayer,
                                                       self.horseshoe_encoder.lambda_a_hat_oplayer,
                                                       self.horseshoe_encoder.lambda_b_hat_oplayer)

        # Compute the entropies
        ent_w, ent_tau, ent_lambda = self.horseshoe_encoder.entropy(sigma, tau_sigma, tau_mu, tau_sigma_global,
                                                                    tau_mu_global,
                                                                    tau_sigma_oplayer, tau_mu_oplayer)

        return log_prior, ent_w, ent_tau, ent_lambda

    def log_likelihood_compute(self, xs_NC, ys_NC, xs_ND, ys_ND, matrix_entries):
        """
        :param xs_NC:
        :param ys_NC:
        :param xs_ND:
        :param ys_ND:
        :param matrix_entries:
        :return:
        """
        n_mc_samples = 20
        log_lik = 0.

        # Generate samples from N(mx_NC, q_sigma) to compute the following
        # E_q[log p(x_ND|mx_NC)]
        for ss in range(n_mc_samples):
            sample_xz_NC = self.draw_sample_from_q(xs_NC)
            sample_xproba_ND = self.decode(sample_xz_NC)

            sample_yz_NC = self.draw_sample_from_q(ys_NC)
            sample_yproba_ND = self.decode(sample_yz_NC)

            # Use MSE to measure reconstruction loss
            # Since MSE is equivalent to log gaussian loss
            log_ll_x_reconstructed = -0.5 * ag_np.sum((sample_xproba_ND - xs_ND) ** 2)
            log_ll_y_reconstructed = -0.5 * ag_np.sum((sample_yproba_ND - ys_ND) ** 2)

            # KL divergence from q(mu, sigma) to prior (std normal)
            log_lik += 1 / n_mc_samples * (log_ll_x_reconstructed + log_ll_y_reconstructed)

        # Compute the loss from adjacency matrix reconstruction
        # Get number of entries
        num_samples = len(matrix_entries)
        f_predict = ag_np.zeros(num_samples)

        # Compute
        # E_q[log p(A_ij|x_i, x_j)] = E_q[Bern(A_ij|sigmoid(x_i dot x_j))]
        for ss in range(n_mc_samples):
            # These two should have the same shape
            # which is (N*C)
            sample_xz_NC = self.draw_sample_from_q(xs_NC)
            sample_yz_NC = self.draw_sample_from_q(ys_NC)

            # inner_prod.shape = (N,)
            inner_prod = ag_np.sum(sample_xz_NC * sample_yz_NC, axis=1)
            f_predict += 1 / n_mc_samples * sigmoid(inner_prod)

        matrix_reconstruction_loss = bce_loss(matrix_entries, f_predict)
        log_lik_matrix_reconstruction = -matrix_reconstruction_loss
        log_lik += log_lik_matrix_reconstruction
        return log_lik

    def decode(self, z):
        return self.decoder.forward(z)

    def draw_sample_from_q(self, xs_NC):
        N = xs_NC.shape[0]
        # The dimension of the code
        C = self.n_dims_code
        # Draw standard normal samples "epsilon"
        # Use the reparameterization trick
        eps_NC = np.random.randn(N, C)
        z_NC = xs_NC + eps_NC * self.q_sigma
        return z_NC

    def calc_vi_loss(self, params, t):
        idx = t % self.M
        # TODO: do mini-batch computation here
        x_ND = self.feature_vectors[idx]
        observed_entries = self.train_adjacency_matrix[idx]
        other_vec_idx = [entry[0][1] for entry in observed_entries]
        matrix_entries = [entry[1] for entry in observed_entries]

        # print("{} {}".format(len(observed_entries), len(matrix_entries)))

        xs_ND = []
        ys_ND = []
        for idy in other_vec_idx:
            xs_ND.append(copy(x_ND))
            ys_ND.append(copy(self.feature_vectors[idy, :]))

        negative_idx = np.random.choice(self.M, self.num_negatives, replace=False)
        for idy in negative_idx:
            xs_ND.append(copy(x_ND))
            ys_ND.append(copy(self.feature_vectors[idy, :]))
            matrix_entries.append(0)


        xs_ND = ag_np.asarray(xs_ND)
        ys_ND = ag_np.asarray(ys_ND)
        matrix_entries = ag_np.asarray(matrix_entries)

        return self.neg_elbo(params, t / self.M, xs_ND, ys_ND, matrix_entries)

    def fit(self, feature_vectors, train_adjacency_matrix, n_epochs=10, l_rate=0.01, test_adjacency_matrix=None, num_negatives=16):
        self.M = feature_vectors.shape[0]
        # TODO: Number of nodes is the number of minibatches in an epoch?
        self.feature_vectors = feature_vectors
        self.train_adjacency_matrix = train_adjacency_matrix
        self.num_negatives = num_negatives

        # TODO: look into how to do mini-batches
        self.test_adjacency_matrix = test_adjacency_matrix

        def callback(params, t, g, decay=0.999):
            # if t % 100 == 0:
            #     print(t)
            if self.polyak:
                # exponential moving average.
                self.polyak_params = decay * self.polyak_params + (1 - decay) * params

            # If empty row in the adjacency matrix, skip this update
            idx = t % self.M
            elbo = -self.calc_vi_loss(params, t)

            self.elbo.append(elbo)

            if (int(t) % self.M) == 0:
                train_err = self.compute_accuracy(params, test=False)
                test_err = self.compute_accuracy(params, test=True)
                self.train_err.append(train_err)
                self.test_err.append(test_err)

                print("Epoch {} elbo {} train-accuracy {} test-accuracy {}".format(t / self.M, self.elbo[-1],
                                                                                       self.train_err[-1],
                                                                                       self.test_err[-1]))

            # if (t % 250) == 0:
            #     # store optimization progress.
            #     self.variational_params_store[t] = copy(params)

            if t > 2:
                if self.elbo[-1] > max(self.elbo[:-1]):
                    self.optimal_elbo_params = copy(params)

            # update inverse gamma distributions
            self.horseshoe_encoder.fixed_point_updates(params)

        # Variational parameters include tau, lambda, v, Vu etc
        init_var_params = self.horseshoe_encoder.initialize_variational_params()
        self.init_params = copy(init_var_params)

        if self.polyak:
            self.polyak_params = copy(init_var_params)

        gradient = grad(self.calc_vi_loss, 0)
        num_iters = n_epochs * self.M  # one iteration = one set of param updates

        # # Run the algorithm using Adam with callback
        # self.variational_params = adam(gradient, init_var_params,
        #                                step_size=0.01, num_iters=num_iters, callback=callback,
        #                                polyak=self.polyak)

        # self.variational_params = rmsprop(gradient, init_var_params,
        #                                    step_size=0.001, gamma=0.9, num_iters=num_iters, callback=callback,
        #                                    polyak=self.polyak)

        self.variational_params = adagrad(gradient, init_var_params,
                                        step_size=0.1, num_iters=num_iters, callback=callback,
                                        polyak=self.polyak)

    def compute_accuracy(self, params, test=True):
        W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
            self.horseshoe_encoder.unpack_params(params)

        z_NC = self.horseshoe_encoder.forward(W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global,
                                              tau_mu_oplayer, tau_sigma_oplayer, self.feature_vectors)

        # z_NC has shape (N,C)
        num_accurate = 0.0
        num_observed = 0.0

        latent_adj_mat = np.dot(z_NC, z_NC.T)

        if test:
            adjancency_matrix = self.test_adjacency_matrix
        else:
            adjancency_matrix = self.train_adjacency_matrix

        for row in adjancency_matrix:
            for coor, val in row:
                if (latent_adj_mat[coor[0]][coor[1]] < 0.0 and val == 0.0 or
                                latent_adj_mat[coor[0]][coor[1]] >= 0.0 and val == 1.0):
                    num_accurate += 1
                num_observed += 1

        return num_accurate / num_observed
