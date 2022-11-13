import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    nweights, nmeans, ncov = estGaussMixEM(ndata, K, n_iter, epsilon)
    sweights, smeans, scov = estGaussMixEM(sdata, K, n_iter, epsilon)
    print(img.shape)

    N = img.shape[0]
    M = img.shape[1]
    
    result = np.ndarray([N, M])
    for i in range(N):
        for j in range(M):
            point = img[i, j, :].reshape(1, img.shape[2])
            sprob = getLogLikelihood(smeans, sweights, scov, point)
            nprob = getLogLikelihood(nmeans, nweights, ncov, point)
            result[i, j] = (sprob/nprob) > theta
    

    return result
