import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    N = samples.shape[0]
    eps = 10e-5
    estDensity = np.ndarray([N, 2])
    for index, x in np.ndenumerate(samples):
        v = 0
        distances = np.abs(samples - x)
        while np.sum(distances < v) < k:
            v += eps
        p = k / (N * 2 * v) # v is the radius
        estDensity[index] = [x, p]
    return estDensity
