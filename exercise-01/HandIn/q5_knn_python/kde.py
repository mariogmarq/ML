import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    N = samples.shape[0]
    estDensity = np.ndarray([N, 2])
    for index, x in np.ndenumerate(samples):
        diference_arr = np.abs(samples - x)
        k = np.sum(gaussian_kernel(diference_arr, h))
        estDensity[index] = [x, (k/N)]

    return estDensity

def gaussian_kernel(u, h):
    return np.exp(
        -(u**2)/(2*h**2)
        )/np.sqrt(2*np.pi*h**2)