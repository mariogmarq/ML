import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    gamma = np.array(gamma)
    X = np.array(X)

    N = gamma.shape[0]
    D = X.shape[1]
    K = gamma.shape[1]

    means = np.ndarray([K, D])
    covariances = np.ndarray([D, D, K])

    # Nj
    n_j = np.ndarray([K])

    for j in range(K):
        n_j[j] = np.sum(gamma[:, j])

    # Weights
    weights = np.ndarray([K])
    for k in range(K):
        weights[k] = n_j[k]/N


    # Means
    for k in range(K):
        acc = 0
        for n in range(N):
            acc += gamma[n, k] * X[n]
        means[k] = acc / n_j[k]

    # Covariances
    for j in range(K):
        acc = np.ndarray([D, D])
        for n in range(N):
            diff = (X[n] - means[j]).reshape(D, 1)
            mat = diff @ diff.T
            acc += gamma[n, j] * mat
        covariances[:, :, j] = acc / n_j[j]
    
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood
