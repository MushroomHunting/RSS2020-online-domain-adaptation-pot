import numpy as np
import matplotlib
import matplotlib.pylab as plt
import ot
import seaborn as sbn
from time import time
from matplotlib import colors as mcolors

from hotpot.chunking import extract_chunk, extract_training_chunk, \
    extract_chunk_subdivs
from hotpot.transfer_datasets import load_learned_data
from hotpot.hilbert_maps import evaluate_model_probs, calc_grid_v2, \
    lidar_hit_miss_masks
from hotpot.random import random_choice_idxs
from hotpot.geometric_masks import bounding_circle_2D

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.6)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

np.random.seed(1111)

if __name__ == "__main__":
    do_append_train_to_test = False
    data_test_newdomain = load_learned_data(dataset="carla_2dtown2_full_auto")
    train_poses_newdomain = data_test_newdomain["train_poses"]
    train_poses_newdomain[:, 0] *= -1
    train_poses_newdomain[:, 2] = data_test_newdomain["train_poses"][:, 3]

    data_test = load_learned_data(dataset="carla_2dtown1_full_auto")
    data = load_learned_data(
        dataset="carla_2dtown1_full_pt1")  # Sparser kernels [Trained kernel data]
    data = {key: data[key] for key in data.keys()}
    data["X_test"] = data_test["X_train"]
    data["Y_test"] = data_test["Y_train"]
    data["train_poses"] = data_test["train_poses"]
    data["test_poses"] = data_test["train_poses"]

    train_poses, test_poses = data["train_poses"], data["test_poses"]
    train_poses[:, 0] *= -1
    test_poses[:, 0] *= -1

    train_poses[:, 2] = train_poses[:, 3]
    test_poses[:, 2] = test_poses[:, 3]

    data["X"] = data["X_test"]
    data["Y"] = data["Y_test"]
    X = data["X"]
    y = data["Y"]
    X_test = data["X_test"]
    N_X_test = X_test.shape[0]
    last_test_timestamp = np.max(data_test_newdomain["X_train"][:, 0])
    y_test = data["Y_test"]

    qhinge_mean = data["qhinge_grid_mean"]
    qgamma_mean_forward = data["qgamma_mean_forward"]
    qw_mean = data["qw_mean"]
    qw_var = data["qw_var"]
    hit_mask, miss_mask = lidar_hit_miss_masks(y=data["Y"])
    hit_mask_test, miss_mask_test = lidar_hit_miss_masks(y=y_test)
    X_hit = X[hit_mask, :]
    y_hit = y[hit_mask]
    X_miss = X[miss_mask, :]
    y_miss = y[miss_mask]

    N_miss_vis = 20000  # This is for faster plotting...
    miss_vis_idxs = random_choice_idxs(X_miss, N=N_miss_vis)
    lidar_radius = 20
    scale_factor = 1 / 50

    colours = list(mcolors.cnames.keys())
    train_chunks = []
    train_chunks_patches = []
    test_chunks = []
    trn_chunk_configs = [[5, np.array([0.1, 0.31])],
                         [5, np.array([0.25, 0.5])],
                         [5, np.array([0.5, 0.75])],
                         [5, np.array([0.9, 1.0])],
                         [20, np.array([0.77, 1.0])],
                         [25, np.array([0.58, 0.85])],
                         ]
    time_idx_trn = 10
    t_trn = np.arange(36)
    t_tst = np.arange(45)
    lidar_search_angle_range = 360
    """-------------------------------------------------|
    | Plot an arbitrary training chunk                  |
    |-------------------------------------------------"""
    chunk, chunk_centroid_x, chunk_centroid_y = extract_training_chunk(
        dataset=data,
        t=t_trn,
        t_idx=time_idx_trn,
        lidar_poses=train_poses,
        lidar_search_angle_range=lidar_search_angle_range,
        search_angle_subdiv=np.array([0.25, 0.5]),
        radius=lidar_radius,
        scale_factor=scale_factor,
        shape="sector")

    THE_FIGURE = plt.figure(1, (17, 9), dpi=200)

    ax = plt.subplot(4, 4, 5)
    if chunk.N_hit > 0:
        ax.scatter(chunk.X_hit_scaled[:, 0], chunk.X_hit_scaled[:, 1],
                   c="darkred",
                   marker='o', s=2)
    if chunk.N_miss > 0:
        ax.scatter(chunk.X_miss_scaled[:, 0], chunk.X_miss_scaled[:, 1],
                   c="darkblue", marker='o', s=1)
    ax.set_title('training chunk LiDAR hits and misses')

    ax = plt.subplot(4, 4, 6)
    if chunk.N_hit > 0:
        ax.scatter(chunk.X_hit_scaled[:, 0], chunk.X_hit_scaled[:, 1],
                   c="darkred",
                   marker='o', s=1)
    scattr = ax.scatter(chunk.qhinge_grid_mean_scaled[:, 0],
                        chunk.qhinge_grid_mean_scaled[:, 1],
                        c=chunk.qgamma_mean_forward.flatten(),
                        marker='o', cmap="rainbow", s=3)
    cbar = plt.colorbar(scattr)
    ax.set_title('t0 corresponding kernels for LiDAR chunk')

    """-------------------------------------------------|
    | Plot the first test lidar scans as from the chunk |
    |-------------------------------------------------"""
    test_chunk1 = extract_chunk(X=X_test,
                                y=y_test,
                                t=np.arange(27),
                                t_idx=5,
                                lidar_poses=test_poses,
                                lidar_search_angle_range=lidar_search_angle_range,
                                search_angle_subdiv=np.array([0.0, 1.0]),
                                radius=lidar_radius,
                                scale_factor=scale_factor,
                                shape="sector")
    ax = plt.subplot(4, 4, 9)
    if test_chunk1.N_hit > 0:
        ax.scatter(test_chunk1.X_hit_scaled[:, 0],
                   test_chunk1.X_hit_scaled[:, 1],
                   c="darkred", marker='o', s=1)
    if test_chunk1.N_miss > 0:
        ax.scatter(test_chunk1.X_miss_scaled[:, 0],
                   test_chunk1.X_miss_scaled[:, 1],
                   c="darkblue", marker='o', s=1)
    ax.set_title('test chunk LiDAR hits and misses')

    """-------------------------------------------------|
    | Create our training chunks                        |
    |-------------------------------------------------"""
    for t_idx, search_angle_subdiv in trn_chunk_configs:
        chunk, chunk_centroid_x, chunk_centroid_y = extract_training_chunk(
            dataset=data,
            t=t_trn,
            t_idx=t_idx,
            lidar_poses=train_poses,
            lidar_search_angle_range=lidar_search_angle_range,
            search_angle_subdiv=search_angle_subdiv,
            radius=lidar_radius,
            scale_factor=scale_factor,
            shape="sector")
        train_chunks.append(chunk)

    do_fusion = False

    do_1st = False
    do_2nd = False
    do_3rd = False
    do_123 = False
    do_all = True

    t_tst = np.arange(int(last_test_timestamp) + 1)

    N_subsample_miss_for_ranking = 50
    N_subsample_for_ranking = 200  # 400
    ranking_ot_reg_e = 0.005  # 0.01
    ranking_ot_metric = "sqeuclidean"
    ranking_ot_tol = 1e-7
    ranking_ot_norm = False  # "max"
    ranking_ot_max_iter = 1000

    N_subsample_for_hq = 1500
    hq_ot_reg_e = 0.001 / 2
    hq_ot_metric = "sqeuclidean"
    if do_all is True:
        hq_verbose = False
    else:
        hq_verbose = True
    hq_ot_max_iter = 2000
    hq_ot_norm = "max"

    subdivs_test = [np.array([0.0, 0.2]),
                    np.array([0.2, 0.4]),
                    np.array([0.4, 0.6]),
                    np.array([0.6, 0.8]),
                    np.array([0.8, 1.0])]

    if do_1st is True:
        q_minmax = (-120.0,  # xmin
                    -60.0,  # xmax
                    180,  # ymin
                    340)
    elif do_2nd is True:
        q_minmax = (-113.0,  # xmin
                    11.0,  # xmax
                    300,  # ymin
                    347)
    elif do_3rd is True:
        q_minmax = (-28.0,  # xmin
                    24.0,  # xmax
                    180,  # ymin
                    340)

    elif do_123 is True:
        q_minmax = (-120.0,  # xmin
                    24.0,  # xmax
                    180,  # ymin
                    347)

    """
    Plot everything!
    """

    ax = plt.subplot(4, 4, 1)
    ax.scatter(X_miss[miss_vis_idxs, 1], X_miss[miss_vis_idxs, 2],
               c="darkblue", marker='o', s=1)
    ax.scatter(X_hit[:, 1], X_hit[:, 2],
               c="darkred", marker='o', s=1)
    ax.set_title('Map LiDAR hits and misses')

    ax = plt.subplot(4, 4, 2)
    ax.scatter(X_hit[:, 1], X_hit[:, 2],
               c="darkred", marker='o', s=3)
    scattr = ax.scatter(qhinge_mean[:, 0], qhinge_mean[:, 1],
                        c=qgamma_mean_forward.flatten(),
                        marker='o', cmap="rainbow", s=3)
    cbar = plt.colorbar(scattr)
    ax.set_title('Kernels with LiDAR hits')

    if do_all is False:
        ax = plt.subplot(4, 4, 3)
        contourf_res = 100
        probs_chunk1, qmesh_x1_chunk1, qmesh_x2_chunk1 = evaluate_model_probs(
            qhinge_grid_mean=qhinge_mean,
            qgamma_mean_forward=qgamma_mean_forward,
            qw_mean=data["qw_mean"],
            qw_var=data["qw_var"],
            qcellres=(1, 1),
            qminmax=q_minmax)  # ymax
        contf = ax.contourf(qmesh_x1_chunk1, qmesh_x2_chunk1,
                            probs_chunk1.reshape(qmesh_x1_chunk1.shape),
                            contourf_res, cmap='jet', vmin=0, vmax=1)
        occupancy_ticks = np.linspace(0.0, 1.0, 5)
        ax.set_title("Predicted Occupancy (Chunk fusion, NAIVE)")
        cbar = plt.colorbar(contf, ticks=occupancy_ticks)

    ax1 = plt.subplot(4, 4, 7)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Lidar scans for each chunk")

    ax2 = plt.subplot(4, 4, 8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Learned kernel gamma for each chunk")

    for i in range(len(train_chunks)):
        plot_offset = np.array(
            np.divmod(i, int(np.sqrt(9)))) * 2 * lidar_radius

        chunk = train_chunks[i]
        if chunk.N_hit > 0:
            ax1.scatter(chunk.X_miss[:, 0] + plot_offset[0],
                        chunk.X_miss[:, 1] + plot_offset[1],
                        c="darkblue", marker='o', s=0.05)
            ax1.scatter(chunk.X_hit[:, 0] + plot_offset[0],
                        chunk.X_hit[:, 1] + plot_offset[1],
                        c="darkred", marker='o', s=1)

        else:
            ax1.scatter(chunk.X_miss[:, 0] + plot_offset[0],
                        chunk.X_miss[:, 1] + plot_offset[1],
                        c="darkblue", marker='o', s=1)
        scattr = ax2.scatter(chunk.qhinge_grid_mean[:, 0] + plot_offset[0],
                             chunk.qhinge_grid_mean[:, 1] + plot_offset[1],
                             c=chunk.qgamma_mean_forward,
                             marker='o', cmap="rainbow", s=2)

    transp_OOS_qhinge_mean_unscaled_FILTERED = np.array([])
    qgamma_mean_forward_FILTERED = np.array([])
    qw_mean_FILTERED = np.array([])
    qw_var_FILTERED = np.array([])
    t_transfer_idxs = np.array([])
    """============================================================|
    |                    Run Sequential Transfer                   |
    |============================================================"""

    used_chunk_idxs = []
    used_chunk_rotations = []
    tic = time()
    for t_idx_test in t_tst:
        print("Progress: {}/{}".format(t_idx_test, t_tst[-1]))
        if do_all is False:
            if do_123 is False:
                if do_1st is True:
                    if t_idx_test > 36:  # First section
                        continue
                elif do_2nd is True:
                    if (t_idx_test < 36) or (t_idx_test > 70):  # Middle section
                        continue
                elif do_3rd is True:
                    if t_idx_test < 65:  # Last one
                        continue

        test_chunk_subdivs = extract_chunk_subdivs(
            X=data_test_newdomain["X_train"],
            y=data_test_newdomain["Y_train"],
            t=t_tst,
            t_idx=t_idx_test,
            lidar_poses=train_poses_newdomain,
            lidar_search_angle_range=lidar_search_angle_range,
            search_angle_subdiv=subdivs_test,
            radius=lidar_radius,
            scale_factor=scale_factor,
            shape="sector")

        for test_chunk in test_chunk_subdivs:
            if test_chunk.N_miss == 0 and test_chunk.N_hit == 0:
                continue  # Bad scan. Abort this scan!
            hitmissdists = []

            for i, train_chunk in enumerate(train_chunks):
                ranking_ot = ot.da.SinkhornTransport(reg_e=ranking_ot_reg_e,
                                                     tol=ranking_ot_tol,
                                                     metric=ranking_ot_metric,
                                                     norm=ranking_ot_norm,
                                                     max_iter=ranking_ot_max_iter,
                                                     limit_max=np.infty,
                                                     verbose=False)

                for j in range(len(train_chunk.R_variations)):
                    # OT distances of misses
                    if test_chunk.N_miss > 0:
                        Xs_miss_scaled = train_chunk.get_variation(
                            "X_miss_scaled", r_idx=j,
                            N_subsample=N_subsample_miss_for_ranking)
                        Xt_miss_scaled = test_chunk.get_variation(
                            "X_miss_scaled", r_idx=0,
                            N_subsample=N_subsample_miss_for_ranking)
                        ranking_ot.fit(Xs=Xs_miss_scaled, Xt=Xt_miss_scaled)
                        miss_w2_dist = np.sum(
                            ranking_ot.cost_ * ranking_ot.coupling_)
                    else:
                        miss_w2_dist = 100.0

                    # OT distances of hits
                    if test_chunk.N_hit > 0 and train_chunk.N_hit > 0:
                        Xs_hit_scaled = train_chunk.get_variation(
                            "X_hit_scaled", r_idx=j,
                            N_subsample=N_subsample_for_ranking)
                        Xt_hit_scaled = test_chunk.get_variation("X_hit_scaled",
                                                                 r_idx=0,
                                                                 N_subsample=N_subsample_for_ranking)

                        try:
                            ranking_ot.fit(Xs=Xs_hit_scaled, Xt=Xt_hit_scaled)
                            hit_w2_dist = np.sum(
                                ranking_ot.cost_ * ranking_ot.coupling_)
                        except:
                            Xs_noise = np.random.normal(0.0, 1e-8,
                                                        size=Xs_hit_scaled.shape)
                            Xt_noise = np.random.normal(0.0, 1e-8,
                                                        size=Xt_hit_scaled.shape)
                            ranking_ot.fit(Xs=Xs_hit_scaled + Xs_noise,
                                           Xt=Xt_hit_scaled + Xt_noise)
                            hit_w2_dist = np.sum(
                                ranking_ot.cost_ * ranking_ot.coupling_)

                    elif test_chunk.N_hit == 0 and test_chunk.N_hit == 0:
                        hit_w2_dist = 1e-8

                    hitmissdists.append((hit_w2_dist, miss_w2_dist, i, j))

            hitmissdists = np.array(hitmissdists)
            max_hitdist = np.max(hitmissdists[:, 0])
            max_missdist = np.max(hitmissdists[:, 1])
            hitmissdists[:, 0] /= max_hitdist
            hitmissdists[:, 1] /= max_missdist
            hitmissdists_sum = np.sum(hitmissdists[:, 0:2], axis=1)
            best_rank_idx = np.argmin(hitmissdists_sum)
            _, _, best_train_chunk_idx, best_r_idx = hitmissdists[best_rank_idx,
                                                     :].astype(np.int)
            used_chunk_idxs.append(best_train_chunk_idx)
            used_chunk_rotations.append(best_r_idx)
            print("best train chunk idx: {}, best r idx: {}".format(
                best_train_chunk_idx, best_r_idx))

            """================================================|
            |             DO THE NEW ATOM TRANSPORT           |
            |================================================"""
            best_chunk = train_chunks[best_train_chunk_idx]
            """ 1. Get the training chunk rotation variation (plot kernels with lengthscale) """
            best_qhinge_grid_mean_original = best_chunk.get_variation(
                "qhinge_grid_mean_scaled", r_idx=best_r_idx, N_subsample=None)
            if test_chunk.N_hit > 0 and best_chunk.N_hit > 0:
                best_X_hit_scaled = best_chunk.get_variation("X_hit_scaled",
                                                             r_idx=best_r_idx,
                                                             N_subsample=N_subsample_for_hq)
                test_X_hit_scaled = test_chunk.get_variation("X_hit_scaled",
                                                             r_idx=0,
                                                             N_subsample=N_subsample_for_hq)
            else:
                best_X_hit_scaled = best_chunk.get_variation("X_miss_scaled",
                                                             r_idx=best_r_idx,
                                                             N_subsample=N_subsample_for_hq)
                test_X_hit_scaled = test_chunk.get_variation("X_miss_scaled",
                                                             r_idx=0,
                                                             N_subsample=N_subsample_for_hq)
            hq_ot = ot.da.SinkhornTransport(reg_e=hq_ot_reg_e,
                                            tol=1e-7,
                                            metric=hq_ot_metric,
                                            norm=hq_ot_norm,
                                            max_iter=hq_ot_max_iter,
                                            limit_max=np.infty,
                                            verbose=hq_verbose)
            try:
                hq_ot.fit(Xs=best_X_hit_scaled, Xt=test_X_hit_scaled)
            except:
                Xs_noise = np.random.normal(0.0, 1e-8,
                                            size=best_X_hit_scaled.shape)
                Xt_noise = np.random.normal(0.0, 1e-8,
                                            size=test_X_hit_scaled.shape)
                hq_ot.fit(Xs=best_X_hit_scaled + Xs_noise,
                          Xt=test_X_hit_scaled + Xt_noise)
            transp_best_X_hit_scaled = hq_ot.transform(Xs=best_X_hit_scaled,
                                                       batch_size=
                                                       best_X_hit_scaled.shape[
                                                           0])
            transp_OOS_qhinge_mean = hq_ot.transform(
                Xs=best_qhinge_grid_mean_original,
                batch_size=best_qhinge_grid_mean_original.shape[0])

            S_unscale = np.identity(n=2) * 1 / train_chunk.scale_factor
            transp_best_X_hit_unscaled = np.dot(transp_best_X_hit_scaled,
                                                S_unscale)
            transp_OOS_qhinge_mean_unscaled = np.dot(transp_OOS_qhinge_mean,
                                                     S_unscale)

            filter_bbox_centroid = test_chunk.get_extraction_bbox_scaled_centroid()
            filter_bbox = bounding_circle_2D(points=transp_OOS_qhinge_mean,
                                             radius=lidar_radius * scale_factor,
                                             centre=filter_bbox_centroid)

            new_transp_OOS_qhinge_mean_FILTERED = transp_OOS_qhinge_mean[
                                                  filter_bbox, :]
            new_qgamma_mean_forward_FILTERED = best_chunk.qgamma_mean_forward[:,
                                               filter_bbox]
            new_qw_mean_FILTERED = best_chunk.qw_mean[filter_bbox, :]
            new_qw_var_FILTERED = best_chunk.qw_var[filter_bbox, :]

            new_transp_OOS_qhinge_mean_unscaled_FILTERED = np.dot(
                new_transp_OOS_qhinge_mean_FILTERED, S_unscale)

            if test_chunk.N_hit > 0:
                hitormiss_centroid_original = test_chunk.hit_centroid_original
            elif test_chunk.N_miss > 0:
                hitormiss_centroid_original = test_chunk.miss_centroid_original

            new_transp_OOS_qhinge_mean_unscaled_FILTERED += np.array(
                hitormiss_centroid_original)
            if transp_OOS_qhinge_mean_unscaled_FILTERED.shape[0] == 0:
                # This is the first iteration. The map is empty.
                transp_OOS_qhinge_mean_unscaled_FILTERED = new_transp_OOS_qhinge_mean_unscaled_FILTERED
                qw_mean_FILTERED = new_qw_mean_FILTERED
                qw_var_FILTERED = new_qw_var_FILTERED
                qgamma_mean_forward_FILTERED = new_qgamma_mean_forward_FILTERED
                t_transfer_idxs = 0 * np.ones(
                    shape=transp_OOS_qhinge_mean_unscaled_FILTERED.shape[0])

            else:
                if do_fusion is False:
                    # Simply concatenate the entire transfer chunk
                    transp_OOS_qhinge_mean_unscaled_FILTERED = np.concatenate(
                        [transp_OOS_qhinge_mean_unscaled_FILTERED,
                         new_transp_OOS_qhinge_mean_unscaled_FILTERED], axis=0)
                    qw_mean_FILTERED = np.concatenate([qw_mean_FILTERED,
                                                       new_qw_mean_FILTERED],
                                                      axis=0)
                    qw_var_FILTERED = np.concatenate([qw_var_FILTERED,
                                                      new_qw_var_FILTERED],
                                                     axis=0)
                    qgamma_mean_forward_FILTERED = np.concatenate(
                        [qgamma_mean_forward_FILTERED,
                         new_qgamma_mean_forward_FILTERED], axis=1)
                    t_transfer_idxs = np.concatenate(
                        [t_transfer_idxs,
                         t_idx_test * np.ones(shape=
                                              new_transp_OOS_qhinge_mean_unscaled_FILTERED.shape[
                                                  0])], axis=0)

    toc = time()
    time_taken = toc - tic
    print("total mapping time: {}".format(time_taken))

    """ ======================================================================
    |   Fuse the transposed filtered chunks and then view the map
    ======================================================================="""

    ax = plt.subplot(4, 4, 15)
    ax.scatter(X_test[hit_mask_test, 1], X_test[hit_mask_test, 2],
               c="darkred", marker='o', s=1)
    scattr = ax.scatter(transp_OOS_qhinge_mean_unscaled_FILTERED[:, 0],
                        transp_OOS_qhinge_mean_unscaled_FILTERED[:, 1],
                        c=qgamma_mean_forward_FILTERED.flatten(),
                        marker='o', cmap="rainbow", s=3)
    cbar = plt.colorbar(scattr)
    ax.set_title('Kernels with LiDAR hits')
    ax.set_xlim((transp_OOS_qhinge_mean_unscaled_FILTERED.min(axis=0)[0],
                 transp_OOS_qhinge_mean_unscaled_FILTERED.max(axis=0)[0]))
    ax.set_ylim((transp_OOS_qhinge_mean_unscaled_FILTERED.min(axis=0)[1],
                 transp_OOS_qhinge_mean_unscaled_FILTERED.max(axis=0)[1]))

    if do_all is False:
        ax = plt.subplot(4, 4, 16)
        probs_chunk1, qmesh_x1_chunk1, qmesh_x2_chunk1 = evaluate_model_probs(
            qhinge_grid_mean=transp_OOS_qhinge_mean_unscaled_FILTERED,
            qgamma_mean_forward=qgamma_mean_forward_FILTERED,
            qw_mean=qw_mean_FILTERED,
            qw_var=qw_var_FILTERED,
            qcellres=(1, 1),
            qminmax=q_minmax)
        contf = ax.contourf(qmesh_x1_chunk1, qmesh_x2_chunk1,
                            probs_chunk1.reshape(qmesh_x1_chunk1.shape),
                            contourf_res, cmap='jet', vmin=0, vmax=1)
        occupancy_ticks = np.linspace(0.0, 1.0, 5)
        ax.set_title("Predicted Occupancy (Chunk fusion, NAIVE)")
        cbar = plt.colorbar(contf, ticks=occupancy_ticks)

    print("used_chunk_idxs: {}".format(used_chunk_idxs))
    print("used_chunk_rotations: {}".format(used_chunk_rotations))

    used_chunk_idxs = np.array(used_chunk_idxs)
    used_chunk_rotations = np.array(used_chunk_rotations)

    train_chunk_idxs = np.arange(len(train_chunks))
    train_chunk_usage_counts = [(used_chunk_idxs == ii).sum() for ii in
                                train_chunk_idxs]
    print("Usage statistics of each training chunk:")
    print(np.array(list(zip(train_chunk_idxs, train_chunk_usage_counts))))

    ## Uncomment if you want to save out the data
    # if do_all is True:
    #     np.savez("carla2dtown1town2_full_auto_SPARSE3a",
    #              qhinge_mean=transp_OOS_qhinge_mean_unscaled_FILTERED,
    #              qw_mean=qw_mean_FILTERED,
    #              qw_var=qw_var_FILTERED,
    #              qgamma_mean_forward=qgamma_mean_forward_FILTERED,
    #              t_transfer_idxs=t_transfer_idxs
    #              )

    plt.tight_layout()
    plt.show()
