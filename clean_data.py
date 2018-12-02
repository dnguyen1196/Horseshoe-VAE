import networkx as nx
import numpy as np
import random
import argparse
import pickle

parser = argparse.ArgumentParser(description="Extract data from KEGG pkl")

parser.add_argument("--feature", type=str, help="File to save feature vector", default="kegg_vector.pkl")
parser.add_argument("--test", type=str, help="File to save test adjancency matrix", default="test_matrix.pkl")
parser.add_argument("--train", type=str, help="File to save train adjancency matrix", default="train_matrix.pkl")

args = parser.parse_args()

feature_file = args.feature
test_file    = args.test
train_file   = args.train

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


pickle.dump(feature_vectors_kegg, open(feature_file, "wb"), protocol=2)
pickle.dump(train_adjacency_matrix_kegg, open(train_file, "wb"), protocol=2)
pickle.dump(test_adjacency_matrix_kegg, open(test_file, "wb"), protocol=2)