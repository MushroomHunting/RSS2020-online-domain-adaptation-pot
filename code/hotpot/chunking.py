import numpy as np
from .math import multi_query_centroids, rot_matrix, get_centroid
from .random import random_choice_idxs
from .geometric_masks import bounding_circle_2D, sector
from .hilbert_maps import lidar_pose, lidar_time_mask, lidar_hit_miss_masks


def multi_query_chunks(X_hit,
                       X_miss,
                       N_query,
                       query_minmax,
                       query_radius,
                       scale_factor,
                       shape="circle"):
    """

    :param X_hit:
    :param X_miss:
    :param N_query:
    :param query_minmax:
    :param query_radius:
    :param scale_factor:
    :param shape:
    :return:
    """
    D = X_hit.shape[1]
    query_centroids = multi_query_centroids(D=D,
                                            N_query=N_query,
                                            query_minmax=query_minmax)

    query_chunks = []
    for i in range(N_query):
        query_centroid = query_centroids[i, :]
        query_chunk = extract_chunk(X_hit=X_hit,
                                    X_miss=X_miss,
                                    centroid=query_centroid,
                                    radius=query_radius,
                                    scale_factor=scale_factor,
                                    shape="circle")
        query_chunks.append(query_chunk)
    return query_chunks


def extract_chunk_subdivs(X, y, t, t_idx, lidar_poses,
                          lidar_search_angle_range, search_angle_subdiv,
                          radius, scale_factor,
                          shape="sector"):
    """

    :param X:
    :param y:
    :param t:
    :param t_idx:
    :param lidar_poses:
    :param lidar_search_angle_range:
    :param search_angle_subdiv:
    :param radius:
    :param scale_factor:
    :param shape:
    :return:
    """
    return [
        extract_chunk(X=X, y=y, t=t, t_idx=t_idx, lidar_poses=lidar_poses,
                      lidar_search_angle_range=lidar_search_angle_range,
                      search_angle_subdiv=subdiv, radius=radius,
                      scale_factor=scale_factor, shape=shape) for subdiv in
        search_angle_subdiv]


class Chunk(object):
    def __init__(self,
                 X_hit,
                 X_miss,
                 N_hit,
                 N_miss,
                 y_hit,
                 y_miss,
                 qw_mean,
                 qw_var,
                 qgamma_mean,
                 qgamma_mean_forward,
                 qgamma_var,
                 qhinge_grid_mean,
                 qhinge_grid_var,
                 scale_factor=1 / 100,
                 lidar_bearing_original=None,
                 extraction_centroid_original=None,
                 extraction_radius_original=None,
                 hit_centroid_original=None,
                 miss_centroid_original=None
                 ):
        """

        :param X_hit:
        :param X_miss:
        :param N_hit:
        :param N_miss:
        :param y_hit:
        :param y_miss:
        :param qw_mean:
        :param qw_var:
        :param qgamma_mean:
        :param qgamma_mean_forward:
        :param qgamma_var:
        :param qhinge_grid_mean:
        :param qhinge_grid_var:
        :param scale_factor:
        :param lidar_bearing_original:
        :param extraction_centroid_original:
        :param extraction_radius_original:
        :param hit_centroid_original:
        :param miss_centroid_original:
        """
        self.scale_factor = scale_factor
        self.S = np.identity(n=2) * self.scale_factor

        self.R_variations = [rot_matrix(degree) for degree in
                             45 * np.arange(0, 5)] + \
                            [rot_matrix(degree) for degree in
                             -45 * np.arange(1, 5)]
        self.X_hit = X_hit  # Use for transportation
        self.X_miss = X_miss  # Use for transportation
        self.N_hit = N_hit
        self.N_miss = N_miss
        if self.N_hit > 0:
            if lidar_bearing_original is not None:
                self.X_hit = np.dot(self.X_hit,
                                    rot_matrix(lidar_bearing_original))
            # Use this normalized one for OT distance
            self.X_hit_scaled = np.dot(self.X_hit, self.S)
        if self.N_miss > 0:
            if lidar_bearing_original is not None:
                self.X_miss = np.dot(self.X_miss,
                                     rot_matrix(lidar_bearing_original))
            # Use this normalized one for OT distance
            self.X_miss_scaled = np.dot(self.X_miss, self.S)
        self.y_hit = y_hit
        self.y_miss = y_miss
        self.qw_mean = qw_mean
        self.qw_var = qw_var
        self.qgamma_mean = qgamma_mean
        self.qgamma_mean_forward = qgamma_mean_forward
        self.qgamma_var = qgamma_var
        self.qhinge_grid_mean = qhinge_grid_mean  # Use for transportation
        if lidar_bearing_original is not None:
            self.qhinge_grid_mean = np.dot(self.qhinge_grid_mean,
                                           rot_matrix(lidar_bearing_original))
        # Use this normalized one for OT distance
        if self.qhinge_grid_mean is not None:
            self.qhinge_grid_mean_scaled = np.dot(self.qhinge_grid_mean, self.S)
        self.qhinge_grid_var = qhinge_grid_var
        # Original centroid of the lidar chunks
        self.extraction_radius_original = extraction_radius_original
        self.extraction_centroid_original = extraction_centroid_original
        if self.extraction_centroid_original is not None:
            self.extraction_centroid_original_scaled = np.dot(
                self.extraction_centroid_original, self.S)
        self.hit_centroid_original = hit_centroid_original
        self.miss_centroid_original = miss_centroid_original
        self.lidar_bearing_original = lidar_bearing_original

    def get_extraction_bbox_scaled_centroid(self):
        """
        Get the original coordinate in scaled space
        :return:
        """
        if self.hit_centroid_original is not None:
            return self.extraction_centroid_original_scaled - np.dot(
                np.array(self.hit_centroid_original), self.S)
        else:
            return self.extraction_centroid_original_scaled - np.dot(
                np.array(self.miss_centroid_original), self.S)

    def get_variation(self, varname="X_hit_scaled", r_idx=1, N_subsample=None):
        """
        :param varname:
        :param r_idx:
        :param N_subsample:
        :return:
        """
        return_val = None
        if varname == "X_hit":
            return_val = np.dot(self.X_hit, self.R_variations[r_idx])
        elif varname == "X_miss":
            return_val = np.dot(self.X_miss, self.R_variations[r_idx])
        elif varname == "X_hit_scaled":
            return_val = np.dot(self.X_hit_scaled, self.R_variations[r_idx])
        elif varname == "X_miss_scaled":
            return_val = np.dot(self.X_miss_scaled, self.R_variations[r_idx])
        elif varname == "qhinge_grid_mean":
            return_val = np.dot(self.qhinge_grid_mean, self.R_variations[r_idx])
        elif varname == "qhinge_grid_mean_scaled":
            return_val = np.dot(self.qhinge_grid_mean_scaled,
                                self.R_variations[r_idx])

        if N_subsample is not None:
            random_idxs = random_choice_idxs(return_val, N_subsample, axis=0)
            return return_val[random_idxs, :]
        else:
            return return_val


def extract_chunk(X,
                  y,
                  t=None,
                  t_idx=None,
                  lidar_poses=None,
                  lidar_search_angle_range=180,
                  search_angle_subdiv=np.array([0.0, 1.0]),
                  radius=40,
                  scale_factor=1 / 100,
                  shape="sector"):
    """

    :param X:
    :param y:
    :param t:
    :param t_idx:
    :param lidar_poses:
    :param lidar_search_angle_range:
    :param search_angle_subdiv:
    :param radius:
    :param scale_factor:
    :param shape:
    :return:
    """
    if shape == "sector":
        pose_t = lidar_pose(lidar_poses, t_idx=t[t_idx])
        bearing_t_deg = np.rad2deg(pose_t[2])
        centroid = pose_t[0:2]
        min_angle_temp = bearing_t_deg - lidar_search_angle_range / 2
        max_angle_temp = bearing_t_deg + lidar_search_angle_range / 2
        angle_range = max_angle_temp - min_angle_temp
        min_angle = min_angle_temp + search_angle_subdiv[0] * angle_range
        max_angle = min_angle_temp + search_angle_subdiv[1] * angle_range
        angle_range = np.array([min_angle,
                                max_angle])

        lidar_mask_t = lidar_time_mask(X, t=t[t_idx])
        X_t = X[lidar_mask_t, :]
        y_t = y[lidar_mask_t]
        hit_mask_t, miss_mask_t = lidar_hit_miss_masks(y=y_t)
        X_hit_t = X_t[hit_mask_t, 1:]  # IMPORTANT TO REMOVE TIMESTAMP
        y_hit_t = y_t[hit_mask_t]
        X_miss_t = X_t[miss_mask_t, 1:]
        y_miss_t = y_t[miss_mask_t]

        X_hit_t_mask = sector(X_hit_t, centroid, radius, angle_range)
        X_miss_t_mask = sector(X_miss_t, centroid, radius, angle_range)

        X_hit_t = X_hit_t[X_hit_t_mask, :]
        X_miss_t = X_miss_t[X_miss_t_mask, :]

        total_hits = X_hit_t.shape[0]
        total_misses = X_miss_t.shape[0]

        X_hit_chunk = None
        X_miss_chunk = None
        hit_centroid_original = None
        miss_centroid_original = None

        chunk_centroid_x, chunk_centroid_y = centroid
        if total_hits > 0:
            hit_centroid_x, hit_centroid_y = hit_centroid_original = get_centroid(
                X_hit_t)
            hit_centroid = np.array([[hit_centroid_x, hit_centroid_y]])
            X_hit_chunk = X_hit_t - hit_centroid

        if total_misses > 0:
            miss_centroid_x, miss_centroid_y = miss_centroid_original = get_centroid(
                X_miss_t)
            miss_centroid = np.array([[miss_centroid_x, miss_centroid_y]])
            if total_hits > 0:
                X_miss_chunk = X_miss_t - hit_centroid
            else:
                X_miss_chunk = X_miss_t - miss_centroid

    chunk = Chunk(X_hit=X_hit_chunk,
                  X_miss=X_miss_chunk,
                  N_hit=total_hits,
                  N_miss=total_misses,
                  y_hit=y_hit_t,
                  y_miss=y_miss_t,
                  qw_mean=None,
                  qw_var=None,
                  qgamma_mean=None,
                  qgamma_mean_forward=None,
                  qgamma_var=None,
                  qhinge_grid_mean=None,
                  qhinge_grid_var=None,
                  scale_factor=scale_factor,
                  lidar_bearing_original=None,
                  extraction_centroid_original=centroid,
                  extraction_radius_original=radius,
                  hit_centroid_original=hit_centroid_original,
                  miss_centroid_original=miss_centroid_original)
    return chunk


def extract_training_chunk(dataset,
                           y_hit=None,
                           y_miss=None,
                           t=None,
                           t_idx=None,
                           lidar_poses=None,
                           lidar_search_angle_range=180,
                           search_angle_subdiv=np.array([0.0, 1.0]),
                           radius=40,
                           scale_factor=1 / 100,
                           shape="sector"):
    """

    :param dataset:
    :param y_hit:
    :param y_miss:
    :param t:
    :param t_idx:
    :param lidar_poses:
    :param lidar_search_angle_range:
    :param search_angle_subdiv:
    :param radius:
    :param scale_factor:
    :param shape:
    :return:
    """
    X = dataset["X"]
    y = dataset["Y"]
    qhinge_mean = dataset["qhinge_grid_mean"]

    if shape == "sector":
        pose_t = lidar_pose(lidar_poses, t_idx=t[t_idx])
        bearing_t_deg = np.rad2deg(pose_t[2])
        centroid = pose_t[0:2]
        min_angle_temp = bearing_t_deg - lidar_search_angle_range / 2
        max_angle_temp = bearing_t_deg + lidar_search_angle_range / 2
        angle_range = max_angle_temp - min_angle_temp
        min_angle = min_angle_temp + search_angle_subdiv[0] * angle_range
        max_angle = min_angle_temp + search_angle_subdiv[1] * angle_range
        angle_range = np.array([min_angle,
                                max_angle])

        lidar_mask_t = lidar_time_mask(X, t=t[t_idx])
        X_t = X[lidar_mask_t, :]
        y_t = y[lidar_mask_t]
        hit_mask_t, miss_mask_t = lidar_hit_miss_masks(y=y_t)
        X_hit_t = X_t[hit_mask_t, 1:]
        y_hit_t = y_t[hit_mask_t]
        X_miss_t = X_t[miss_mask_t, 1:]
        y_miss_t = y_t[miss_mask_t]

        X_hit_t_mask = sector(X_hit_t, centroid, radius, angle_range)
        X_miss_t_mask = sector(X_miss_t, centroid, radius, angle_range)

        X_hit_t = X_hit_t[X_hit_t_mask, :]
        X_miss_t = X_miss_t[X_miss_t_mask, :]

        total_hits = X_hit_t_mask.sum()
        total_misses = X_miss_t_mask.sum()

        qhinge_mean_mask = sector(qhinge_mean, centroid, radius, angle_range)
        qhinge_mean_t = qhinge_mean[qhinge_mean_mask, :]
        qgamma_mean_forward_t = dataset["qgamma_mean_forward"][:,
                                qhinge_mean_mask]
        qw_mean_t = dataset["qw_mean"][qhinge_mean_mask, :]
        qw_var_t = dataset["qw_var"][qhinge_mean_mask, :]

        X_hit_chunk = None
        X_miss_chunk = None
        hit_centroid_original = None
        miss_centroid_original = None

        chunk_centroid_x, chunk_centroid_y = centroid
        if total_hits > 0:
            hit_centroid_x, hit_centroid_y = hit_centroid_original = get_centroid(
                X_hit_t)

            hit_centroid = np.array([[hit_centroid_x, hit_centroid_y]])
            X_hit_chunk = X_hit_t - hit_centroid

        if total_misses > 0:
            miss_centroid_x, miss_centroid_y = miss_centroid_original = get_centroid(
                X_miss_t)
            miss_centroid = np.array([[miss_centroid_x, miss_centroid_y]])
            if total_hits > 0:
                # Prioritise the lidar hit centroid over the miss centroid
                X_miss_chunk = X_miss_t - hit_centroid
            else:
                # X_miss_chunk = X_miss[bbox_X_miss, :] - miss_centroid
                X_miss_chunk = X_miss_t - miss_centroid

        if total_hits > 0:
            qhinge_grid_mean_chunk = qhinge_mean_t - hit_centroid
        else:
            qhinge_grid_mean_chunk = qhinge_mean_t - miss_centroid

    new_chunk = Chunk(X_hit=X_hit_chunk,
                      X_miss=X_miss_chunk,
                      N_hit=total_hits,
                      N_miss=total_misses,
                      y_hit=y_hit,
                      y_miss=y_miss,
                      qw_mean=qw_mean_t,
                      qw_var=qw_var_t,
                      qgamma_mean=None,
                      qgamma_mean_forward=qgamma_mean_forward_t,
                      qgamma_var=None,
                      qhinge_grid_mean=qhinge_grid_mean_chunk,
                      qhinge_grid_var=None,
                      scale_factor=scale_factor,
                      lidar_bearing_original=bearing_t_deg,
                      extraction_centroid_original=centroid,
                      extraction_radius_original=radius,
                      hit_centroid_original=hit_centroid_original,
                      miss_centroid_original=miss_centroid_original)
    return new_chunk, chunk_centroid_x, chunk_centroid_y
