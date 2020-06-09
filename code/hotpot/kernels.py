import numpy as np

from .math import paired_squared_euclidean


def rbf_kernel(X, Y, gamma, bias=False):
    """
    rbf features (hinged rbf when Y are grid locations)
    :param X:
    :param Y:
    :param gamma:
    :param tfdt:
    :return:
    """
    batch_size = X.shape[0]
    dist = np.exp(-gamma * paired_squared_euclidean(X, Y))
    if bias is False:
        features = dist
    else:
        features = np.concatenate(
            [np.ones(shape=(batch_size, 1)), dist], axis=1)
    return features
