import numpy as np


def q_matrix(dist):
    y_dist = dist
    qji = 1 / (1 + y_dist)
    # print(qji)
    np.fill_diagonal(qji, 0)
    sum_qi = np.sum(qji, axis=1)
    # sum_qi = np.reshape(sum_qi, [-1, 1])
    # print(sum_qi.shape)
    # print(sum_qi)

    return qji / sum_qi

