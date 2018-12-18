import argparse
from hs_vae.autoencoder import *
from hs_vae.horseshoe_autoencoder import *
import pickle
import numpy

parser = argparse.ArgumentParser(description="Testing VAE")

parser.add_argument("--vae", type=str, help="Types of VAE (standard, laplace or horseshoe prior)", choices=["S", "L", "HS"])
parser.add_argument("--nns", nargs='+', help="Neural network structure, default = [300]", default=[300])
parser.add_argument("--ncode", type=int, help="Number of hidden dimension, default = 100", default=100)
parser.add_argument("--data", type=str, help="Dataset to use, synthetic or kegg dataset", choices=["synthetic", "kegg"])

args = parser.parse_args()

vae = args.vae
nns = [int(x) for x in args.nns]
ncode = args.ncode
data  = args.data


if data == "synthetic":
    # Do synthetic data testing
    f = 5 # Number of fake dims
    num_nodes = 1000
    total_dim = 25
    observed_dim = total_dim - f
    true_dim = 10
    num_fake_dim = f
    hidden_layer_sizes = nns

    n_epochs = 151 if vae != "HS" else 16

    sparsity = 0.25
    # NOTE that there exists the 'interchangeability problem'
    # XR^TRX = X^TX
    # Y = XR^T
    true_vectors, feature_vectors, train_adjacency_matrix, test_adjacency_matrix \
        = create_synthetic_data(num_nodes, sparsity, true_dim, observed_dim, f)

    if vae == "L" or vae == "S":
        l = 1e-3 if vae == "L" else 0 # Regularization term
        vae_model = VariationalAutoencoder(n_dims_code=ncode, \
                                           n_dims_data=observed_dim + num_fake_dim, \
                                           hidden_layer_sizes=hidden_layer_sizes, \
                                           reg_lambda=l)

    else:
        vae_model = HS_VAE(q_sigma=0.25,
                n_dims_code=ncode,
                n_dims_data=total_dim,
                hidden_layer_sizes=hidden_layer_sizes,
                classification=True,
                batch_size=128,
                lambda_b_global=1.0,
                warm_up=False,
                polyak=False)

    vae_model.fit(feature_vectors, train_adjacency_matrix, n_epochs=n_epochs, test_adjacency_matrix=test_adjacency_matrix, num_negatives=0)

    # Pickle the model after training to plot weights
    if vae == "HS":
        param_file = "HS-param-synthetic.pkl"
        invgamma_file = "HS-horseshoe.pkl"
        with open(param_file, "wb") as f:
            pickle.dump(vae_model.optimal_elbo_params, f)
            f.close()
        with open(invgamma_file, "wb") as f:
            pickle.dump(vae_model.horseshoe_encoder, f)
            f.close()

    else: # Need to pickle the params because cannot pickle lambdas
        with open("{}-synthetic.pkl".format(vae), "wb") as f:
            pickle.dump(vae_model.encoder.params, f)


else: # Do Kegg Data testing

    # feature_vectors_kegg = pickle.load(open("kegg_vector.pkl", "rb"))
    # test_adjacency_matrix_kegg = pickle.load(open("test_matrix.pkl", "rb"))
    # train_adjacency_matrix_kegg = pickle.load(open("train_matrix.pkl", "rb"))

    import networkx as nx
    import numpy as np
    import random

    random.seed(42)
    G = nx.read_gpickle("./data/kegg.ungraph.pkl")

    node_to_ind = {}
    ind_to_node = {}
    ind = 0
    all_chem = []
    feature_vectors_kegg = []
    adjacency_matrix_kegg = []
    unique_links_list = []
    adjacency_list_kegg = {}
    non_edges = set()
    num_edges = 0
    neg_ratio = 1.0

    # Create feature vectors
    for n in G.nodes:
        all_chem.append(n)
        node_to_ind[n] = ind
        ind_to_node[ind] = n
        ind += 1
        feature_vectors_kegg.append(G.nodes[n]["fingerprint"].astype(float))

    feature_vectors_kegg = np.vstack(feature_vectors_kegg)

    # Record positive edges into adjacency set
    for e in G.edges:
        num_edges += 1
        if e[0] not in adjacency_list_kegg:
            adjacency_list_kegg[e[0]] = set()
        if e[1] not in adjacency_list_kegg:
            adjacency_list_kegg[e[1]] = set()
        adjacency_list_kegg[e[0]].add(e[1])
        adjacency_list_kegg[e[1]].add(e[0])
        unique_links_list.append(([node_to_ind[e[0]], node_to_ind[e[1]]], 1))

    # Randomly sample non-edges with neg_ratio * amount as edges
    num_sampled = 0
    while num_sampled < neg_ratio * num_edges:
        v1 = random.randint(0, len(all_chem) - 1)
        v2 = random.randint(0, len(all_chem) - 1)
        n1 = ind_to_node[v1]
        n2 = ind_to_node[v2]

        if (
                            (n1, n2) not in non_edges and
                            (n2, n1) not in non_edges and
                        n2 not in adjacency_list_kegg[n1]
        ):
            non_edges.add((n1, n2))
            unique_links_list.append(([v1, v2], 0))

        num_sampled += 1


    def create_train_test_adj_matrices(unique_links, num_nodes, ratio=0.8):
        num_train_links = int(ratio * len(unique_links))
        train_inds = random.sample(range(0, len(unique_links)), num_train_links)
        test_inds = list(set(range(0, len(unique_links))) - set(train_inds))
        train_links = [unique_links[i] for i in train_inds]
        test_links = [unique_links[i] for i in test_inds]

        train_sparse_matrix = [[] for _ in range(num_nodes)]
        test_sparse_matrix = [[] for _ in range(num_nodes)]

        for ([id1, id2], val) in test_links:
            test_sparse_matrix[id1].append(([id1, id2], val))
            if id1 != id2:
                test_sparse_matrix[id2].append(([id2, id1], val))

        for ([id1, id2], val) in train_links:
            train_sparse_matrix[id1].append(([id1, id2], val))
            if id1 != id2:
                train_sparse_matrix[id2].append(([id2, id1], val))

        return train_sparse_matrix, test_sparse_matrix


    train_adjacency_matrix_kegg, test_adjacency_matrix_kegg = create_train_test_adj_matrices(unique_links_list,
                                                                                             len(all_chem), ratio=0.8)

    if vae == "L" or vae == "S":
        l = 1e-3 if vae == "L" else 0 # Regularization term
        vae_model = VariationalAutoencoder(n_dims_code=ncode, \
                                           n_dims_data=feature_vectors_kegg.shape[1], \
                                           hidden_layer_sizes=nns,\
                                           reg_lambda=l)
        vae_model.fit(feature_vectors_kegg, train_adjacency_matrix_kegg, n_epochs=151, test_adjacency_matrix=test_adjacency_matrix_kegg)

    else:
        vae_model = HS_VAE(q_sigma=0.2,
            n_dims_code=20,
            n_dims_data=feature_vectors_kegg.shape[1],
            hidden_layer_sizes=nns,
            classification=True,
            batch_size=64,
            lambda_b_global=1.0,
            warm_up=False,
            polyak=False)

        num_negatives = 16
        vae_model.fit(feature_vectors_kegg, train_adjacency_matrix_kegg, n_epochs=16, test_adjacency_matrix=test_adjacency_matrix_kegg, num_negatives=num_negatives)
