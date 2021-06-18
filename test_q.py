import numpy as np
from test_euc import euc_x
from test_exp import produce_p_matrix


def q_matrix(dist):
    y_dist = dist
    qji = (1 + y_dist) ** -1
    # print(qji)
    np.fill_diagonal(qji, 0)
    sum_qi = np.sum((qji), axis=1)
    sum_qi = np.reshape(sum_qi, [-1, 1])
    # print(sum_qi.shape)
    # print(sum_qi)

    return qji / sum_qi


def main():
    Y = np.random.randn(7,2)
    Y = euc_x(Y)
    y_prob = q_matrix(Y)


if __name__ == '__main__':
    main()
#
#
# x1 = [2,5,1,3]
# x2 = [4,2,9,0]
# x3 = [4,3,2,1]
# x4 = [1,5,3,4]
# x5 = [3,6,2,8]
# x6 = [7,4,2,9]
# x7 = [6,2,4,5]
#
# X = np.array([x1,x2,x3,x4,x5,x6,x7])
# print(X)
# x_dist = euc_x(X)
# p = produce_p_matrix(x_dist, 3)
# print(p)
#
# y = np.random.randn(len(X),2)
# print(y)
# y_dist = euc_x(y)
# print(y_dist)
# Q = q_matrix(y_dist)
# print("Q is:", Q)
#
# P = 0.5 * (p + p.T)
# print("P is:", P)
#
# grads = np.zeros([len(y), 1])
# print(grads)
# Prob_diffs = P - Q
# Y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0)
