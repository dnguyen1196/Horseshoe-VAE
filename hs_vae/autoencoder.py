from __future__ import print_function

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from data.utils import *
from hs_vae.horseshoe_autoencoder import *
import autograd.numpy as np
from builtins import range

"""
Simple neural network class
"""

class NeuralNetwork(nn.Module):
    def __init__(
            self,
            n_dims_code=16,
            n_dims_data=32,
            hidden_layer_sizes=[32],
            encoder=True,
    ):

        """
        q_sigma = 0.2
        """
        super(NeuralNetwork, self).__init__()

        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
        )
        self.n_layers = len(layer_sizes) - 1

        if not encoder:
            layer_sizes = [a for a in reversed(layer_sizes)]

        self.activations = list()
        self.params = nn.ModuleList()
        for (n_in, n_out) in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.params.append(nn.Linear(n_in, n_out))
            self.activations.append(F.relu)
        self.activations[-1] = lambda a: a

    def forward(self, x):
        # Note that if x contains multiple instance
        # if x.shape = (num_sample, in_dim)
        # then the output shape will be (num_sample, out_dim)
        cur_arr = x
        for ll in range(self.n_layers):
            linear_func = self.params[ll]
            a_func = self.activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        mu_NC = cur_arr
        return mu_NC

    def regularize(self):
        reg_loss = 0.0
        for param in self.params:
            mats = torch.cat([x.view(-1) for x in param.parameters()])
            reg_loss += torch.norm(mats, 1)

        return reg_loss


"""
Standard variational auto-encoder
"""


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            q_sigma=0.2,
            n_dims_code=16,
            n_dims_data=64,
            hidden_layer_sizes=[32],
            reg_lambda=0.0,
    ):

        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.q_sigma = torch.Tensor([float(q_sigma)])
        self.lam = reg_lambda

        self.epsilon = 1

        # Encoder network
        self.encoder = NeuralNetwork(
            n_dims_code=n_dims_code,
            n_dims_data=n_dims_data,
            hidden_layer_sizes=hidden_layer_sizes,
            encoder=True,
        )
        # Decoder network
        self.decoder = NeuralNetwork(
            n_dims_code=n_dims_code,
            n_dims_data=n_dims_data,
            hidden_layer_sizes=hidden_layer_sizes,
            encoder=False,
        )
        self.train_losses = []
        self.test_losses = []
        self.ELBOs = []
        self.epochs = []
        self.all_epochs = []

    def elbo_epochs(self):
        return self.all_epochs

    def loss_epochs(self):
        return self.epochs

    def elbos(self):
        return self.ELBOs

    def tr_losses(self):
        return self.train_losses

    def te_losses(self):
        return self.test_losses

    def forward(self, x_ND):
        """
        Run entire probabilistic autoencoder on input (encode then decode)

        Returns
        -------
        xproba_ND : 1D array, size of x_ND
        """
        mu_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC)
        return self.decode(z_NC), mu_NC

    def draw_sample_from_q(self, mu_NC):
        ''' Draw sample from the probabilistic encoder q(z|mu(x), \sigma)

        We assume that "q" is Normal with:
        * mean mu (argument of this function)
        * stddev q_sigma (attribute of this class, use self.q_sigma)

        Args
        ----
        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.

        Returns
        -------
        z_NC : tensor-like, N x C
            Exactly one sample vector for each of the N images in minibatch.
        '''
        # Number of samples
        N = mu_NC.shape[0]

        # The dimension of the code
        C = self.n_dims_code

        if self.training:
            # Draw standard normal samples "epsilon"
            # Use the reparameterization trick
            eps_NC = torch.randn(N, C)
            z_NC = mu_NC + eps_NC * self.q_sigma
            return z_NC
        else:
            # For evaluations, we always just use the mean
            return mu_NC

    def encode(self, x_ND):
        """
        Args
        ----
        x_ND: the observation vector
        """
        return self.encoder.forward(x_ND)

    def decode(self, z_NC):
        """
        Args
        ----
        z_NC: the code vector
        """
        return self.decoder.forward(z_NC)

    def binary_predict_error_rate(self, f_predict, f_true):
        """

        """
        length = f_predict.size()[0]
        f_predict_binary = (f_predict > 0.5).type(torch.FloatTensor)
        error = torch.sum((f_predict_binary - f_true).abs_()) / length
        return error

    def calc_vi_loss(self, xs_ND, ys_ND, vals, n_mc_samples=1):
        """
        Args

        xs_ND: the input feature vectors
        ys_ND: the other feature vectors in mini-batch
        vals: the entry associated with (x,y)
        n_mc_samples:

        ----
        Returns:
        loss

        """
        neg_expected_ll = 0.0
        epsilon = 0.0001 # Small number to avoid numerical problem

        # Given a (potentially) a tensor of observation vectors,
        # Encode it into latent space
        mx_NC = self.encode(xs_ND)
        my_NC = self.encode(ys_ND)

        # Compute the KL divergence
        # KL(N(mx_NC, q_sigma) || N(0, I))
        kl_xz_NC = -0.5 * torch.sum(1 + torch.log(self.q_sigma ** 2) - mx_NC ** 2 - self.q_sigma ** 2)
        kl_yz_NC = -0.5 * torch.sum(1 + torch.log(self.q_sigma ** 2) - my_NC ** 2 - self.q_sigma ** 2)
        kl = kl_xz_NC + kl_yz_NC  # Total KL term

        # Generate samples from N(mx_NC, q_sigma) to compute the following
        # E_q[log p(x_ND|mx_NC)]
        # for ss in range(n_mc_samples):
        #     sample_z_NC = self.draw_sample_from_q(mx_NC)
        #     sample_xproba_ND = self.decode(sample_z_NC)
        #
        #     # Use MSE to measure reconstruction loss
        #     # Since MSE is equivalent to log gaussian loss
        #     sample_mse_loss = F.mse_loss(sample_xproba_ND, xs_ND)
        #
        #     # KL divergence from q(mu, sigma) to prior (std normal)
        #     # Have to multiply (1 - epsilon) to make sure all entries are < 1
        #     neg_expected_ll += 1 / n_mc_samples * sample_mse_loss

        # Compute the loss from adjacency matrix reconstruction
        # Get number of entries
        num_samples = len(vals)
        f_predict = torch.zeros(num_samples)

        # Compute
        # E_q[log p(A_ij|x_i, x_j)] = E_q[Bern(A_ij|sigmoid(x_i dot x_j))]

        for ss in range(n_mc_samples):
            # These two should have the same shape
            # which is (N*C)
            sample_z_NC = self.draw_sample_from_q(mx_NC)
            sample_y_NC = self.draw_sample_from_q(my_NC)

            # inner_prod.shape = (N,)
            inner_prod = torch.sum(sample_z_NC * sample_y_NC, dim=1)
            f_predict += 1 / n_mc_samples * torch.sigmoid(inner_prod) * (1 - epsilon)

        # Use binary cross entry loss, NOTE that this is for
        # adjacency matrix whose entry values are 0 and 1
        # This will need to change for other types of adjacency matrix value
        # Use the binary prediction loss with logits
        matrix_reconstruction_loss = \
            F.binary_cross_entropy(f_predict, Variable(torch.FloatTensor(vals)))

        neg_expected_ll += matrix_reconstruction_loss

        # L1 regularizer term
        reg_loss = self.lam * (self.encoder.regularize() + self.decoder.regularize())

        return neg_expected_ll, kl, reg_loss, matrix_reconstruction_loss

    def fit(self, feature_vectors, train_adjacency_matrix, n_epochs=10, test_adjacency_matrix=None, num_negatives=16):
        # Initialize optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        optimizer = torch.optim.Adagrad(self.parameters(), lr=0.1)
        num_nodes = feature_vectors.shape[0]
        self.num_negatives = num_negatives

        for epoch in range(n_epochs):
            # Do round-robin optimization
            for idx in range(num_nodes):
                x_ND = Variable(torch.FloatTensor([feature_vectors[idx, :]]))
                ys_ND = list()
                observed_entries = train_adjacency_matrix[idx]
                other_vec_idx = [entry[0][1] for entry in observed_entries]
                vals = [entry[1] for entry in observed_entries]

                for idy in other_vec_idx:
                    ys_ND.append(feature_vectors[idy, :])

                negative_idx = np.random.choice(num_nodes, self.num_negatives, replace=False)
                for idy in negative_idx:
                    ys_ND.append(copy(feature_vectors[idy, :]))
                    vals.append(0)

                ys_ND = Variable(torch.FloatTensor(ys_ND))
                optimizer.zero_grad()

                # NOTE:
                # expected_ll refers to the expected log likelihood term of ELBO
                # kl refers to the KL divergence term of the ELBO
                # matrix_loss refers to the matrix reconstruction loss
                neg_expected_ll, KL, reg, matrix_loss = self.calc_vi_loss(x_ND, ys_ND, vals, n_mc_samples=10)

                KL = 1 / len(vals) * KL
                # TODO: scale the KL term
                # ELBO loss = negative expected log likelihood + KL
                elbo_loss = neg_expected_ll + KL + reg

                elbo_loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                all_vectors = Variable(torch.FloatTensor(feature_vectors))
                train_accuracy = classification_accuracy(self, all_vectors, train_adjacency_matrix)
                test_accuracy = classification_accuracy(self, all_vectors, test_adjacency_matrix)
                self.epochs.append(epoch)
                self.all_epochs.append(epoch)
                self.ELBOs.append(-elbo_loss)
                self.train_losses.append(1.0 - train_accuracy)
                self.test_losses.append(1.0 - test_accuracy)

                #             if epoch in [0, 1, 25] or epoch % 50 == 0:
                #                 all_vectors = Variable(torch.FloatTensor(feature_vectors))
                #                 train_accuracy = classification_accuracy(self, all_vectors, train_adjacency_matrix)
                #                 test_accuracy = classification_accuracy(self, all_vectors, test_adjacency_matrix)
                #                 self.epochs.append(epoch)
                #                 self.train_losses.append(1.0 - train_accuracy)
                #                 self.test_losses.append(1.0 - test_accuracy)

                print("epoch: ", epoch, " - objective loss: ", np.around(elbo_loss.data.item(), 4),
                      " - train accuracy: ",
                      np.around(train_accuracy, 4), " - test accuracy: ", np.around(test_accuracy, 4))



