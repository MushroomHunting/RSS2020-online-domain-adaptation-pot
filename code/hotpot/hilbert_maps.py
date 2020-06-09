import numpy as np
from scipy.special import expit
from .kernels import rbf_kernel
from tqdm import tqdm


def lidar_hit_miss_masks(y, miss_value=0, hit_value=1):
    """

    :param y:
    :param miss_value:
    :param hit_value:
    :return:
    """
    hit_mask = (y == hit_value).flatten()
    miss_mask = (y == miss_value).flatten()
    return hit_mask, miss_mask


def lidar_time_mask(X, t):
    """

    :param X:
    :param t:
    :return:
    """
    return X[:, 0] == t


def lidar_pose(poses, t_idx):
    """
    Utility function for returning the poses associated with a particular timestamp
    :param poses:
    :param t_idx:
    :return:
    """
    return poses[t_idx, :]


def calc_grid_v2(cell_resolution, max_min, method='grid', X=None, M=None):
    """
    :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
    :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
    :param X: a sample of lidar locations
    :return: numpy array of size (# of RNFs, 2) with grid locations
    """
    if max_min is None:
        # if 'max_min' is not given, make a boundarary based on X
        # assume 'X' contains samples from the entire area
        expansion_coef = 1.2
        x_min, x_max = expansion_coef * X[:, 0].min(), expansion_coef * X[:,
                                                                        0].max()
        y_min, y_max = expansion_coef * X[:, 1].min(), expansion_coef * X[:,
                                                                        1].max()
    else:
        x_min, x_max = max_min[0], max_min[1]
        y_min, y_max = max_min[2], max_min[3]

    if method == 'grid':  # on a regular grid
        xvals = np.arange(x_min, x_max, cell_resolution[0])
        yvals = np.arange(y_min, y_max, cell_resolution[1])
        xx, yy = np.meshgrid(xvals, yvals)
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    else:  # sampling
        D = 2
        if M is None:
            xsize = np.int((x_max - x_min) / cell_resolution[0])
            ysize = np.int((y_max - y_min) / cell_resolution[1])
            M = np.int((x_max - x_min) / cell_resolution[0]) * np.int(
                (y_max - y_min) / cell_resolution[1])
        if method == 'mc':
            grid = np.random.uniform(0, 1, (M, D))
        else:
            grid = None

        grid[:, 0] = x_min + (x_max - x_min) * grid[:, 0]
        grid[:, 1] = y_min + (y_max - y_min) * grid[:, 1]

    return grid, xx, yy


def eval_probs_only(X_q,
                    qhinge_grid_mean,
                    qgamma_mean_forward,
                    qw_mean,
                    qw_var,
                    ):
    """

    :param X_q:
    :param qhinge_grid_mean:
    :param qgamma_mean_forward:
    :param qw_mean:
    :param qw_var:
    :return:
    """
    X_q_features = rbf_kernel(X_q, qhinge_grid_mean, qgamma_mean_forward)
    post_mu = np.dot(X_q_features, qw_mean)
    post_var = np.sum(np.square(X_q_features) * np.transpose(qw_var), axis=1, keepdims=True)
    kappa_var = 1 / np.sqrt(1 + np.pi * post_var / 8)
    probs = expit(kappa_var * post_mu)
    return probs


def evaluate_model_probs(qhinge_grid_mean,
                         qgamma_mean_forward,
                         qw_mean,
                         qw_var,
                         qcellres=(4, 4),
                         qminmax=(0, 300, 50, 250),
                         ):
    """

    :param qhinge_grid_mean:
    :param qgamma_mean_forward:
    :param qw_mean:
    :param qw_var:
    :param qcellres:
    :param qminmax:
    :return:
    """
    X_q, qmesh_x1, qmesh_x2 = calc_grid_v2(qcellres, qminmax, method='grid', X=None)
    X_q_features = rbf_kernel(X_q, qhinge_grid_mean, qgamma_mean_forward)
    post_mu = np.dot(X_q_features, qw_mean)
    post_var = np.sum(np.square(X_q_features) * np.transpose(qw_var), axis=1, keepdims=True)
    kappa_var = 1 / np.sqrt(1 + np.pi * post_var / 8)
    probs = expit(kappa_var * post_mu)
    return probs, qmesh_x1, qmesh_x2


def evaluate_model_probs_batched(qhinge_grid_mean,
                                 qgamma_mean_forward,
                                 qw_mean,
                                 qw_var,
                                 # overflow_offset=10,
                                 batch_size=500,  # batch size in x and y directios
                                 qcellres=(4, 4),
                                 qminmax=(0, 300, 50, 250),
                                 ):
    """

    :param qhinge_grid_mean:
    :param qgamma_mean_forward:
    :param qw_mean:
    :param qw_var:
    :param batch_size:
    :param qcellres:
    :param qminmax:
    :return:
    """
    X_q, qmesh_x1, qmesh_x2 = calc_grid_v2(qcellres, qminmax, method='grid', X=None)
    N_fullbatch, size_lastbatch = np.divmod(X_q.shape[0], batch_size)
    probs = np.array([])

    # Process the full query area chunk by chunk
    # This can be clearly be parallelized if you wanted
    total_iters = N_fullbatch + int(size_lastbatch > 0)
    for i in tqdm(range(total_iters)):
        if i < N_fullbatch:
            x_idx_start = i * batch_size
            x_idx_end = x_idx_start + batch_size
            X_q_batch = X_q[x_idx_start:x_idx_end, :]
        else:  # The remainder batch
            x_idx_start = N_fullbatch * batch_size
            X_q_batch = X_q[x_idx_start:, :]

        # Now we're going to extract kernels relevant only in the vicinity of the query
        X_q_features = rbf_kernel(X_q_batch, qhinge_grid_mean, qgamma_mean_forward)
        post_mu = np.dot(X_q_features, qw_mean)
        post_var = np.sum(np.square(X_q_features) * np.transpose(qw_var), axis=1, keepdims=True)
        kappa_var = 1 / np.sqrt(1 + np.pi * post_var / 8)
        probs = np.concatenate([probs, expit(kappa_var * post_mu).flatten()])

    return probs, qmesh_x1, qmesh_x2
