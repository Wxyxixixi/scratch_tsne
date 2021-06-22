import numpy as np


def euc_x(X):
    # L2 distance : (a-b)^2 = a^2 + b^2 -2ab
    sq_X = np.sum(np.square(X),axis = 1)
    sq_X = np.reshape(sq_X, [-1, 1])
    mul = np.dot(X, X.T)
    result = np.add(np.add(sq_X, -2*mul), sq_X.T)
    return result





