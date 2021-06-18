import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from test_euc import euc_x
from test_exp import produce_p_matrix
from test_q import q_matrix
import matplotlib.pyplot as plt


def main():
    digits = load_digits()
    x = digits.data
    label = digits.target

    # pca_x = reduce_dims_pca(x)
    # loading data into distance
    dist = euc_x(x)
    PERPLEXITY = 30
    # creates matrix P from them using gaussian kernel and update it by set perplexity
    P_matrix = produce_p_matrix(dist, PERPLEXITY)
    P = (P_matrix + np.transpose(P_matrix)) / 2
    # print(P_joint)
    Y = np.random.random([dist.shape[0], 2])   # initial Y matrix

    num_iter = 500
    lr = 0.1
    for i in range(num_iter):
        Y_dist = euc_x(Y)
        # print(Y_dist)
        Q = q_matrix(Y_dist)
        if (i + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (i + 1), ": error is ", C)
        grades = CalculateGradients(P,Q,Y,Y_dist)
        # print(grades)


        # update Y
        Y = Y + lr * grades

    data_2d = Y.copy()
    print(Y.shape)
    print(Y)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c = label)
    plt.show()


def CalculateGradients(P_matrix, Q_matrix, Y_point,y_dist):
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





if __name__ == "__main__":
    main()










