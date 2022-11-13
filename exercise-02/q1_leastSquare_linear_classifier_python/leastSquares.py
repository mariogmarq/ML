import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    
    #PfdLinearDiscrimintans,slide 33

    #Dimensions
    dim = data.shape[1]
    N = data.shape[0]

    X_ext = np.c_[np.ones(N) , data]
    
    #W = np.linalg.inv(X.T @ X) @ X.T @ label
    W = label.T @ (np.linalg.pinv(X_ext)).T

    weight = W[1:]
    bias = W[0]
    
    return weight, bias
