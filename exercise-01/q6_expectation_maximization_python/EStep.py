import numpy as np
from getLogLikelihood import gaussianProb, getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    means = np.array(means)
    covariances = np.array(covariances)
    weights = np.array(weights)
    X = np.array(X)

    N = X.shape[0]
    K = means.shape[0]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    gamma = np.ndarray([N, K])

    for i in range(N):
        denominador = 0
        for k in range(K):
            denominador += weights[k] * gaussianProb(
                X[i], means[k], covariances[:, :, k]
            )
        for j in range(K):
            numerador = weights[j] * gaussianProb(X[i], means[j], covariances[:, :, j])
            gamma[i, j] = numerador / denominador

    return [logLikelihood, gamma]
