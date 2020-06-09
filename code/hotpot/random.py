import numpy as np

def random_choice_idxs(input, N, axis=0):
    N = np.minimum(input.shape[axis], N)
    return np.random.choice(input.shape[axis], N, replace=False)