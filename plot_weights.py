import matplotlib.pyplot as plt
import seaborn as sb
import argparse
from sklearn.externals import joblib
import pickle
from hs_vae.autoencoder import *
from hs_vae.horseshoe_autoencoder import *

sb.set_context("paper", rc={"lines.linewidth": 5, "lines.markersize":10, 'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'ylabel.fontsize':15,
   'xlabel.fontsize':15,
   'text.usetex': False,
    'axes.titlesize' : 25,
    'axes.labelsize' : 25,  })
sb.set_style("darkgrid")


def plot_singlelayer_weights(horseshoe_encoder, optimal_elbo_params):
    plt.figure()
    axx = plt.gca()

    w_vect, sigma_vect, tau_mu_vect, tau_sigma_vect, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
            horseshoe_encoder.unpack_params(optimal_elbo_params)

    # Show only the input layer
    for layer_id, (mu, var, tau_mu, tau_sigma) in enumerate(zip(horseshoe_encoder.unpack_layer_weights(w_vect),
                                                                horseshoe_encoder.unpack_layer_weights(
                                                                    sigma_vect),
                                                                horseshoe_encoder.unpack_layer_weight_priors(
                                                                    tau_mu_vect),
                                                                horseshoe_encoder.unpack_layer_weight_priors(
                                                                    tau_sigma_vect))):
        scale_mu = 0.5 * (tau_mu + tau_mu_global[layer_id])
        scale_v = 0.25 * (tau_sigma ** 2 + tau_sigma_global[layer_id] ** 2)
        w, b = mu

        wstack = np.vstack([w, b]) * np.exp(scale_mu - np.sqrt(scale_v))

        # idx = np.argsort(np.linalg.norm(wstack, axis=0))
        # if idx.shape[0] > 20:
        #     sb.boxplot(data=wstack[:, idx[-20:]], orient="h", ax=axx)
        # else:
        #     sb.boxplot(data=wstack[:, idx], orient="h", ax=axx)

        sb.boxplot(data=wstack, orient="h", ax=axx)

        plt.show(block=True)
        return wstack


# Note the model type
model_type = "S"


if model_type == "HS":
    param_file = "HS-param-synthetic.pkl"
    invgamma_file = "HS-horseshoe.pkl"

    with open(param_file, "rb") as f1:
        elbo_param= pickle.load(f1)
    with open(invgamma_file, "rb") as f:
        invgamma = pickle.load(f)
    plot_singlelayer_weights(invgamma, elbo_param)

else:
    param_file = "{}-synthetic.pkl".format(model_type)
    with open(param_file, "rb") as f:
        params = pickle.load(f)
        inputlayer = params[0].weight
        print(inputlayer)
