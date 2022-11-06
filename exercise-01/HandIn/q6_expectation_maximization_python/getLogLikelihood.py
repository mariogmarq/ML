import numpy as np


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####

    logLikelihood = 0
    covariances = np.array(covariances)
    means = np.array(means)

    X = np.array(X)
    K = covariances.shape[2]
    N = X.shape[0]

    for n in range(N):
        acc = 0
        for k in range(K):
            acc += weights[k] * gaussianProb(X[n, :], means[k], covariances[:, :, k])
        logLikelihood += np.log(acc)

    return logLikelihood


def gaussianProb(X, mean, covariance):
    D = len(mean)
    div = np.float_power(2 * np.pi, D / 2.0) * np.sqrt(np.linalg.norm(covariance))
    inv = np.linalg.inv(covariance)
    diff = (X-mean)
    e = diff.T @ inv
    e = e @ diff
    e = np.exp(-0.5 * (X - mean).T @ np.linalg.inv(covariance) @ (X - mean))
    return e / div
