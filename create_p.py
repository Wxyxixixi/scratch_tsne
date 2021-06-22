import numpy as np


def produce_p_matrix(distance, PERPLEXITY):
    dist = distance
    sigmas = np.ones(np.shape(dist[0])) # initial sigma as [1,1,1,1,1,1,1]
    sigmas = np.reshape(sigmas, [-1, 1])
    sigmas = sigmas * 500
    perp = PERPLEXITY

    P_initial = CalculateProb(dist, sigmas)
    Perplexities = CalculatePerplexities(P_initial)
    P_update = np.zeros(np.shape(P_initial))
    perp_update = []
    for i in range(len(dist)):
        diff = Perplexities[i] - perp
        upper = 1000
        lower = 1e-10
        guess = sigmas[i]
        tol = 1e-4
        tries = 0
        while abs(diff) > tol and tries < 50:
            if diff < 0:
                lower = guess
                guess = 0.5 * (upper + guess)
            else:
                upper = guess
                guess = 0.5 * (lower + guess)
            this_prob = CalculateProbi(dist[i], guess, i)
            this_prob = np.maximum(this_prob, 1e-12)
            this_perp = CalculatePerplexity(this_prob, i)
            diff = this_perp - perp
            tries += 1

        P_update[i] = this_prob
        perp_update.append(this_perp)
        sigmas[i] = guess
    print('perplexity:', perp_update)

    return P_update


def CalculateProbi(dist, sigma, idx):
    Pi = np.exp(- dist / (2 * (sigma ** 2)))
    Pi[idx] = 0
    sum_Pi = np.sum(Pi)
    return Pi / sum_Pi


def CalculateProb(dist, sigmas):
    P = np.exp(- dist / (2 * (sigmas ** 2)))
    np.fill_diagonal(P, 0)  # do not care about i == j
    sum_P = np.sum(P, axis=1)
    sum_P = np.reshape(sum_P, [-1, 1])
    P_initial = P / sum_P  ## normalize P

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

