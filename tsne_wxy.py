import numpy as np
from sklearn.decomposition import PCA

from load_data import dataLoader
from create_distance import euc_x
from create_p import produce_p_matrix
from create_q import q_matrix

import matplotlib.pyplot as plt
import time


def CalculateGradients(P_matrix, Q_matrix, Y_point, y_dist):
    pq_diff = P_matrix - Q_matrix
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y_point, 1) - np.expand_dims(Y_point, 0)
    # print(y_diffs)
    distance_inv = (1 + y_dist) ** -1
    distance_expanded = np.expand_dims(distance_inv, 2)
    # print(distsss_expanded)
    results = pq_expanded * y_diffs * distance_expanded
    grades = np.sum(results, axis=1)
    return grades


def reduce_dims_pca(x, n_components=30):
    """for faster implementation, first use pca to reduce dimensions. Nx30
    """
    pca = PCA(n_components=n_components)
    pca.fit(x.T)
    return pca.components_.T


def tsne(x, perplexity=30, num_iter=500, lr=1):

    # loading data into distance
    print("Computing pairwise distances...")
    dist = euc_x(x)
    print('dist:',dist)

    initial_momentum = 0.5
    final_momentum = 0.8
    # creates matrix P from them using gaussian kernel and update it by set perplexity
    P_matrix = produce_p_matrix(dist, perplexity)
    P = (P_matrix + np.transpose(P_matrix)) / 2
    np.fill_diagonal(P, 1e-12)
    Y = np.random.randn(dist.shape[0], 2)   # initial Y matrix

    Y_m2 = Y.copy()
    Y_m1 = Y.copy()

    print("start fitting tsne...")
    print("T-SNE DURING:%s" % time.perf_counter())
    # c = []
    for i in range(num_iter):
        Y_dist = euc_x(Y)
        # print(Y_dist)
        Q = q_matrix(Y_dist)
        # print(Q)
        np.fill_diagonal(Q, 1.)
        if (i + 1) % 10 == 0:
            # np.fill_diagonal(Q, 1e-12)
            C = np.sum(np.multiply(P, np.log(P / Q)))
            C /= np.shape(Q[0])
            oldC = C
            ratio = C / oldC
            # c.append(c)

            print("Iteration ", (i + 1), ": error is ", C)
            print("ratio is ", ratio)
            # if (i +1) != 10:
            #     ratio = C/oldC
            #     print("ratio ", ratio)
            #     # if ratio >= 0.95:
            #     #     break
            # oldC = C
            # np.fill_diagonal(Q, 1e-12)
        grads = CalculateGradients(P, Q, Y, Y_dist)
        # print(grades)
        # update Y
        Y = Y - lr * grads
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        Y += momentum * (Y_m1 - Y_m2)
        # update previous Y's for momentum
        Y_m2 = Y_m1.copy()
        Y_m1 = Y.copy()

    data = Y.copy()
    print("finished training!")
    return data


if __name__ == "__main__":
    data, label = dataLoader()
    # x = reduce_dims_pca(data)
    data_2d = tsne(data)
    y = np.linspace(0, 500, 10)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=label)
    plt.show()




