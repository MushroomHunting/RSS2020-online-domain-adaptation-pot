import numpy as np

def is_in_hull(P, hull):
    """
    Determine if the list of points P lies inside the hull

    Here we use the equations of the plan to determine if the point is outside the hull.
    We never build the Delaunay object.
    This function takes as input P, a (m,n) array of m points in n dimensions. The hull is constrcuted using

    ref: https://stackoverflow.com/a/52405173
    :param P:
    :param hull:
    :return: list
             List of boolean where true means that the point is inside the convex hull
    """

    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    in_hull_mask = np.all((A @ np.transpose(P)) <= np.tile(-b, (1, len(P))),
                          axis=0)
    return in_hull_mask