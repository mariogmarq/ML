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
    estDensity = np.ndarray([N, 2])
    for index, x in np.ndenumerate(samples):
        distances = np.sort(np.abs(samples - x))
        v = 2 * np.max(distances[:k+1])
        estDensity[index] = [samples[index], k / (N * v)]

    return estDensity
