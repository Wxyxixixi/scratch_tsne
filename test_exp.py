import numpy as np
from sklearn.datasets import load_digits
from test_euc import euc_x


def produce_p_matrix(distance, PERPLEXITY):
    dist = distance
    #dist = Distance
    # print(dist)
    sigmas = np.ones(np.shape(dist[0])) # initial sigma as [1,1,1,1,1,1,1]
    sigmas = np.reshape(sigmas,[-1,1])
    sigmas = sigmas * 500
    # print("sigma is:", sigmas)
    perp = PERPLEXITY

    P_initial = CalculateProb(dist, sigmas)
    Perplexities = CalculatePerplexities(P_initial)
    P_update = np.zeros(np.shape(P_initial))
    perp_update = []
    # sigmas_update = []
    for i in range(len(dist)):
        # print(i)
        diff = Perplexities[i] - perp
        upper = 1000
        lower = 1e-10
        guess = sigmas[i]
        tol = 1e-4
        # print("inital diff is:", diff)
        while abs(diff) > tol:
            if diff < 0:
                lower = guess
                guess = 0.5 * (upper + guess)
            else:
                upper = guess
                guess = 0.5 * (lower + guess)
            # print(guess)
            this_prob = CalculateProbi(dist[i], guess, i)
            # print("this prob:", this_prob)
            this_perp = CalculatePerplexity(this_prob, i)
            diff = this_perp - perp
            # print("_______diff________")
            # print(diff)
            # print("____________________")
        P_update[i] = this_prob
        perp_update.append(this_perp)
        sigmas[i] = guess
    # print(P_update)
    # print(perp_update)
    # print(sigmas)
    return P_update


def CalculateProbi(dist, sigma, idx):
    Pi = np.exp(- dist / (2 * (sigma ** 2)))
    # print("Pi is :", Pi)
    Pi[idx] = 0
    # print("Pi is :", Pi)
    sum_Pi = np.sum(Pi)
    # print(Pi / sum_Pi)
    return Pi / sum_Pi



def CalculateProb(dist, sigmas):
    P = np.exp(- dist / (2 * (sigmas ** 2)))
    # print((2 * (sigma ** 2)))
    np.fill_diagonal(P, 0)  # do not care about i == j
    # print(P)
    sum_P = np.sum(P, axis=1)
    sum_P = np.reshape(sum_P, [-1, 1])
    # print(sum_P)
    P_initial = P / sum_P  ## normalize P
    # print("*************************")
    # print(P_initial)
    # print(P_initial.shape)
    # print("*************************")
    return P_initial


def CalculatePerplexity(p_i, idx):
    pi = p_i.copy()
    pi[idx] = 1
    entropy =  - np.sum(np.multiply(pi, np.log2(pi)))
    perp = 2 ** entropy
    return perp


def CalculatePerplexities(P):
    # entropy = np.zeros([1,np.shape(P[0]))
    np.fill_diagonal(P, 1)
    # print(P)
    entropy = -np.sum(np.multiply(P, np.log2(P)), axis=1)
    entropy = np.reshape(entropy, [-1,1])
    # print("entropy:", entropy)
    perp = 2 ** entropy
    # print("perp:", perp)
    return perp

#
# if __name__ == "__main__":
#     distance = euc_x()
#     ProducePMatrix(distance,30)