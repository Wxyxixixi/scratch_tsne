import numpy as np
from sklearn.datasets import load_digits

# def euc_x1_x2(x1,x2):
#     return np.sum(np.square(x1) + np.square(x2) - 2*np.multiply(x1, x2))


def euc_x(X):
    sq_X = np.sum(np.square(X),axis = 1)
    sq_X = np.reshape(sq_X, [-1, 1])
    mul = np.dot(X, X.T)
    # np.fill_diagonal(mul, 0)
    #
    # print(sq_X)
    # print(sq_X.T)
    # print(mul)
    result = np.add(np.add(sq_X, -2*mul),sq_X.T)
    # np.fill_diagonal(result,0)
    return result




# if __name__ == '__main__':
#     Distance()
