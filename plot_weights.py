import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sb
import argparse
from sklearn.externals import joblib
from statistics import mean
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

        wstack = np.vstack([w]) * np.exp(scale_mu - np.sqrt(scale_v))

        w_mean = [np.average(np.abs(wstack[i])) for i in range(len(wstack))]
        w_std = [np.std(np.abs(wstack[i])) for i in range(len(wstack))]

        plt.errorbar(range(15), w_mean, w_std, linestyle='None', marker='^')
        locator = matplotlib.ticker.MultipleLocator(2)
        plt.gca().xaxis.set_major_locator(locator)
        formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.show()

        print("Feature index sorted by weight magnitude in increasing order")
        idx = np.argsort(np.linalg.norm(wstack.T, axis=0))
        print(idx)

        sb.boxplot(data=wstack, orient="h", ax=axx)

        return wstack


# Note the model type
model_type = "HS"


if model_type == "HS":
    param_file = "HS-param-synthetic.pkl"
    invgamma_file = "HS-horseshoe.pkl"

    with open(param_file, "rb") as f1:
        elbo_param = pickle.load(f1)
    with open(invgamma_file, "rb") as f:
        invgamma = pickle.load(f)
    plot_singlelayer_weights(invgamma, elbo_param)

else:
    param_file = "{}-synthetic.pkl".format(model_type)
    with open(param_file, "rb") as f:
        params = pickle.load(f)
        inputlayer = params[0].weight
        print(inputlayer)
