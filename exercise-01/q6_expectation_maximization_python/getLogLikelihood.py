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
    X = np.array(X)
    k = covariances.shape[2]
    means = np.array(means)
    for n in range(X.shape[0]):
        acc = 0
        for j in range(k):
            acc += weights[j] * gaussianProb(X[n], means[j], covariances[:, :, j])
        logLikelihood += np.log(acc)

    return logLikelihood


def gaussianProb(X, mean, covariance):
    D = len(mean)
    div = np.float_power(2 * np.pi, D / 2.0) * np.sqrt(np.linalg.norm(covariance))
    e = np.exp(-0.5 * (X - mean).T @ np.linalg.inv(covariance) @ (X - mean))
    return e/div