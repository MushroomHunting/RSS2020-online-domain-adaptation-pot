import numpy as np
import scipy.stats as sstats
import ghalton as gh

def rot_matrix(ccw_deg=0.0):
    theta = np.radians(ccw_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R.T


def paired_squared_euclidean(A, B):
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    return np.square(A_norm) - \
           2 * np.dot(A, B.T) + \
           np.square(np.transpose(B_norm))


def get_centroid(X):
    """
    Find the centre of a set of coordinates
    :param X:
    :return:
    """
    centre = np.mean(X, axis=0)
    return centre[0], centre[1]


def xyspan(X):
    min_x = X[:, 0].min()
    max_x = X[:, 0].max()
    min_y = X[:, 1].min()
    max_y = X[:, 1].max()

    x_range = max_x - min_x
    y_range = max_y - min_y

    return x_range, y_range

def multi_query_centroids(D,
                          N_query,
                          query_minmax):
    sequencer = gh.GeneralizedHalton(gh.EA_PERMS[:D])
    points = np.array(sequencer.get(int(N_query)))
    query_xmin, query_xmax, query_ymin, query_ymax = query_minmax  # (150, 200, 150, 300)
    query_centroids = sstats.uniform.ppf(points,
                                         loc=(query_xmin, query_ymin),
                                         scale=(query_xmax - query_xmin,
                                                query_ymax - query_ymin))
    return query_centroids
