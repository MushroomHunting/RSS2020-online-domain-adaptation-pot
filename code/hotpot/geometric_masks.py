import numpy as np
from .math import rot_matrix


def sector(points, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.

    adapted from ref: https://stackoverflow.com/a/18354475
    :param points:
    :param centre:
    :param radius:
    :param angle_range:  (start, stop)
    :return:
    """
    x, y = points[:, 0], points[:, 1]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)
    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi
    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)
    # circular mask
    circmask = r2 <= radius * radius
    # angular mask
    anglemask = theta <= (tmax - tmin)
    return circmask * anglemask


def uniform_in_rectangle(origin=np.array([[0.0, 0.0]]), length=10.0,
                         height=10.0,
                         cw_deg=0.0, N=100):
    pts = np.random.uniform((0.0, 0.0), (length, height), size=(N, 2))
    R = rot_matrix(cw_deg)
    pts = pts[np.random.choice(pts.shape[0], N, replace=False)]
    return origin + np.dot(pts, R)


def uniform_on_rectangle(origin=np.array([[0.0, 0.0]]), length=10.0,
                         height=10.0,
                         cw_deg=0.0, N=100):
    pts_N = np.random.uniform((0.0, height), (length, height), size=(N, 2))
    pts_S = np.random.uniform((0.0, 0.0), (length, 0.0), size=(N, 2))
    pts_E = np.random.uniform((length, 0.0), (length, height), size=(N, 2))
    pts_W = np.random.uniform((0.0, 0.0), (0.0, height), size=(N, 2))
    pts = np.concatenate([pts_N, pts_E, pts_S, pts_W])
    R = rot_matrix(cw_deg)
    pts = pts[np.random.choice(pts.shape[0], N, replace=False)]
    return origin + np.dot(pts, R)


def bounding_box_2D(points,
                    min_x=-np.inf, max_x=np.inf,
                    min_y=-np.inf, max_y=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------
    points: (n,2) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1],
                ...,
                [xn,yn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be kept or not.
        The size of the boolean mask will be the same as the number of given points.

    Ref: https://github.com/daavoo/pyntcloud
    """
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bb_filter = np.logical_and(bound_x, bound_y)
    return bb_filter


def bounding_circle_2D(points,
                       radius,
                       centre):
    dists = np.linalg.norm(points - centre, axis=1)
    mask = (dists <= radius)
    return mask
