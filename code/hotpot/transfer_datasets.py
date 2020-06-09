import os.path
import numpy as np


def load_learned_data(dataset):
    """
    -------------------- Keys --------------------

    'X',                    -   Training lidar scan coordinates
    'Y',                    -   Lidar hit value 0 or 1
    'Y_q',                  -
    'X_test',               -   Query locations
    'Y_test',               -
    'qw_mean',              -   (M, 1)
                                Model weight means
    'qw_var'                -   (M, 1)
                                Model weight variances
    'qgamma_mean'           -   (1, M)
                                Model gamma means
    'qgamma_mean_forward'   -   (1, M)
                                Model gamma means with bijection forward pass
    'qgamma_var'            -   (1, M)
                                Model gamma variances
    'qhinge_grid_mean'      -   (M, 2)
                                Model kernel hinge position means (learned)
    'qhinge_grid_var'       -   (M, 2)
                                Model kernel hinge position variances (learned)
    ----------------------------------------------
    :param dataset:
    :return:
    """

    if dataset == "carla_2dtown":  # This is the 3 part streaming dataset
        filename = "carla_2d_town1.npz"
    elif dataset == "carla_2dtown_c":  # This has the trained kernels (very dense)
        filename = "carla2dtown_BBVI.npz"
    elif dataset == "carla_2dtown1_full":  # This is the full 5 part
        filename = "carla_2d_town1_full.npz"
    # carla2d TOWN #1
    elif dataset == "carla_2dtown1_full_auto":  # This is the full 5 part
        filename = "carla_2d_town1_full_auto.npz"
    elif dataset == "carla_2dtown1_full_pt1":  # This has the trained kernels
        filename = "carla2dtown1full_pt1_BBVI.npz"
    # carla2d TOWN #2
    elif dataset == "carla_2dtown2_full_auto":  # This is the full 5 part
        filename = "carla_2d_town2_full_auto.npz"
    elif dataset == "carla_2dtown2_full_pt1":  # This has the trained kernels
        filename = "carla2dtown2fullauto_pt1_BBVI.npz"
    elif dataset == "carla_2dtown2_full_pt2":  # This has the trained kernels
        filename = "carla2dtown2fullauto_pt2_BBVI.npz"
    else:
        raise ValueError("Hey! Invalid dataset name :(")
    this_file_path = os.path.dirname(__file__)
    dataset_path = os.path.join(this_file_path,
                                "..",
                                "datasets",
                                filename)
    data = np.load(dataset_path)
    return data
