import numpy as np
import torch
import autograd.numpy as ag_np

def compute_entry_values(coordinates, latent_vectors):
    """
    coordinates: list of coordinates

    """
    vals = list()
    for i, j in coordinates:
        if np.sum(latent_vectors[i] * latent_vectors[j]) > 0:
            vals.append(1)
        else:
            vals.append(0)
    return vals


def create_sparse_matrix_list(observed_coordinates, num_nodes, latent_vectors):
    """
    observed_coordinates: list of (row,col) coordinates
    num_nodes : int, number of nodes
    latent_vectors : latent vectos whose inner products used to determine adjacency
    matrix

    ---
    return the sparse representation of the adjacency matrix

    """
    sparse_adjacency_matrix = [[] for _ in range(num_nodes)]

    for entry in observed_coordinates:
        idx, idy = entry[0], entry[1]
        inner = np.dot(latent_vectors[idx, :], latent_vectors[idy, :])
        val = 0
        if inner > 0:
            val = 1

        sparse_adjacency_matrix[idx].append((entry, val))
        sparse_adjacency_matrix[idy].append(([idy, idx], val))

    return sparse_adjacency_matrix


def create_observed_features(latent_vectors, num_nodes, true_dim, observed_dim, noise_feature_num=0):
    """
    latent_vectors : shape(num_nodes, true_dim), the latent feature vectors of all nodes
    num_nodes : int, number of nodes in this network
    true_dim  : int, dimension of the latent feature (the code)
    observed_dim : int, dimension of the observed feature vector
    num_noisy_dim : int, added noisy dimension

    ---

    Create observed features that contain both true and noisy features

    1. Create a random transformation matrix A
    2. Apply the transformation A X_l where X_l is the latent feature vectors
    3.
    ---
    returns

    augmented_feature_matrix : shape(num_nodes, observed_dim + num_noisy_dim)
    the augmented observed feature vector for all the nodes, with relevent
    dimensions together with noisy entries.

    """

    # Create a random transformation, A
    transformation_matrix = np.random.randn(observed_dim, true_dim)

    # X_true = dot(A, x_true)
    transformed_feature = np.dot(transformation_matrix, latent_vectors.T)
    transformed_feature = transformed_feature.T
    transformed_feature /= transformed_feature.max()  # Scale the feature vectors by dividing by the max value

    # If no noisy feature, just return the transformed feature
    if noise_feature_num == 0:
        return transformed_feature

    # If there are noisy features, generate these from standard normal distribution
    # Noise feature = N(0, I)
    # Create noise matrix X_noise
    noise_feature_matrix = np.random.randn(num_nodes, noise_feature_num)
    augmented_dim = observed_dim + noise_feature_num

    # Horizontally concatenate X_true :: X_noise
    augmented_feature_matrix = np.hstack((transformed_feature, noise_feature_matrix))

    # Permute the features column (Comment out to make all the 'fake' features to concatenate to the end of true features)
    # augmented_feature_matrix = augmented_feature_matrix[:, np.random.permutation(augmented_dim)]
    return augmented_feature_matrix


def create_synthetic_data(num_nodes, sparsity, true_dim, observed_dim, num_noisy_dim):
    """
    num_nodes : int, number of nodes in this network
    sparsity  : float, ratio of all entries, which are observed
    true_dim  : int, dimension of the latent feature (the code)
    observed_dim : int, dimension of the observed feature vector
    num_noisy_dim : int, added noisy dimension

    ---
    returns

    (latent_vectors, observed_feature_vectors, sparse_adjancency_matrix)

    latent_vectors : shape(num_nodes, true_dim), all the latent feature vectors
    observed_feature_vectors : shape(num_nodes, observed_dim), all the observed feature vectors
    sparse_adjancency_matrix : list[list(entry, value)]

    """
    # Create num_nodes hidden vector randomly
    latent_vectors = np.random.multivariate_normal(np.zeros((true_dim,)), \
                                                   np.eye(true_dim), \
                                                   size=(num_nodes,))

    coordinates = [[x, y] for x in range(num_nodes) for y in range(x + 1, num_nodes)]
    total_num_pairs = len(coordinates)
    num_observed = int(len(coordinates) * sparsity)

    # Pick a number of random coordinates
    observed_idx = np.random.choice(total_num_pairs, num_observed, replace=False)
    observed_coordinates = np.take(coordinates, observed_idx, axis=0)

    train_size = int(len(observed_coordinates) * 0.8)
    train_coordinates = observed_coordinates[:train_size]
    test_coordinates = observed_coordinates[train_size + 1:]

    # Create sparse matrix representation
    train_sparse_adjacency_matrix = create_sparse_matrix_list(train_coordinates, num_nodes, latent_vectors)
    test_sparse_adjacency_matrix = create_sparse_matrix_list(test_coordinates, num_nodes, latent_vectors)

    # Create test matrix and train matrix
    observed_feature_vectors = create_observed_features(latent_vectors, num_nodes, true_dim, observed_dim,
                                                        num_noisy_dim)

    return latent_vectors, observed_feature_vectors, train_sparse_adjacency_matrix, test_sparse_adjacency_matrix


def classification_accuracy(model, feature_tensor, adjacency_matrix, report=False):
    """
    Feature vectors.shape -> (num_nodes, number of observed features)
    adjancency_matrix -> sparse representation of adjancey matrix
    where each row -> list of (coordinates, value) associated with the specific
    factor

    Uses model.encode(feature_vector)
    """
    latent_vectors = model.encode(feature_tensor)
    latent_adj_mat = torch.mm(latent_vectors, latent_vectors.transpose(0 , 1))
    num_accurate = 0.0
    num_observed = 0.0
    for row in adjacency_matrix:
        for coor, val in row:
            if (
                latent_adj_mat[coor[0]][coor[1]].item() < 0.0 and val == 0.0 or
                latent_adj_mat[coor[0]][coor[1]].item() >= 0.0 and val == 1.0
            ):

                if report:
                    print("predict: ", latent_adj_mat[coor[0]][coor[1]].item(), " val: ", val)

                num_accurate += 1
            num_observed += 1

    return num_accurate/ num_observed


def classification_error_rate(model, feature_tensor, adjacency_matrix, report=False):
    return 1  -classification_accuracy(model, feature_tensor, adjacency_matrix, report)


def bce_loss(x, x_prime):
    temp1 = x * ag_np.log(x_prime + 1e-10)
    temp2 = (1 - x) * ag_np.log(1 - x_prime + 1e-10)
    bce = -ag_np.sum(temp1 + temp2)
    return bce


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+ag_np.exp(-x))