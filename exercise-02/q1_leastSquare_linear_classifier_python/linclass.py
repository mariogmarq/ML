import numpy as np

def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction
    N = data.shape[0]
    class_pred = np.array([weight @ data[i, :] + bias for i in range(N)])
    class_pred = np.array([1 if x > 0 else -1 for x in class_pred]) # Put between +-1

    return class_pred


