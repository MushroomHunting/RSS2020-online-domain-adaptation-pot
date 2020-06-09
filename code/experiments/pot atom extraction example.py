import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sbn
from matplotlib import colors as mcolors

from hotpot.chunking import extract_chunk, extract_training_chunk
from hotpot.transfer_datasets import load_learned_data
from hotpot.hilbert_maps import evaluate_model_probs, \
    lidar_hit_miss_masks
from hotpot.random import random_choice_idxs

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.6)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

np.random.seed(1)

if __name__ == "__main__":
    do_append_train_to_test = False
    data_test = load_learned_data(dataset="carla_2dtown")
    # Load the learned kernels file
    data = load_learned_data(dataset="carla_2dtown_c")
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
    last_test_timestamp = np.max(X_test[:, 0])
    y_test = data["Y_test"]

    if do_append_train_to_test is True:
        test_poses = np.concatenate([test_poses, train_poses], axis=0)
        X_test = np.concatenate([X_test, X], axis=0)
        X_test[N_X_test:, 0] += last_test_timestamp + 1
        N_X_test = X_test.shape[0]
        last_test_timestamp = np.max(X_test[:, 0])
        y_test = np.concatenate([y_test, y], axis=0)
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
    trn_chunk_configs = [[5, np.array([0.0, 0.33])],
                         [5, np.array([0.33, 0.66])],
                         [5, np.array([0.66, 1.0])],
                         ]

    time_idx_trn = 10
    t_trn = np.arange(36)
    t_tst = np.arange(100)
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

    THE_FIGURE = plt.figure(1, (15, 4), dpi=200)
    """-------------------------------------------------|
    | Plot the first test lidar scans as from the chunk |
    |-------------------------------------------------"""
    test_chunk1 = extract_chunk(X=X_test,
                                y=y_test,
                                t=np.arange(27),
                                t_idx=0,
                                lidar_poses=test_poses,
                                lidar_search_angle_range=lidar_search_angle_range,
                                search_angle_subdiv=np.array([0.0, 1.0]),
                                radius=lidar_radius,
                                scale_factor=scale_factor,
                                shape="sector")

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

    do_1st = True
    do_2nd = False
    do_3rd = False
    do_123 = True

    N_subsample_miss_for_ranking = 50
    N_subsample_for_ranking = 200  # 400
    ranking_ot_reg_e = 0.005  # 0.01
    ranking_ot_metric = "sqeuclidean"
    ranking_ot_tol = 1e-7
    ranking_ot_norm = False
    ranking_ot_max_iter = 1000

    N_subsample_for_hq = 1500
    hq_ot_reg_e = 0.001
    hq_ot_metric = "sqeuclidean"

    hq_ot_max_iter = 2000
    hq_ot_norm = "max"
    barycenter_metric = "sqeuclidean"

    subdivs_test = [np.array([0.0, 0.2]),
                    np.array([0.2, 0.4]),
                    np.array([0.4, 0.6]),
                    np.array([0.6, 0.8]),
                    np.array([0.8, 1.0])]

    if do_1st is True:
        q_minmax = (-120.0,  # xmin
                    -90.0,  # xmax
                    212,  # ymin
                    241)
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

    qgamma_mean_forward_plot = np.clip(qgamma_mean_forward, 0.0, 0.6)

    ax = plt.subplot(1, 5, 1)
    ax.scatter(X_miss[miss_vis_idxs, 1], X_miss[miss_vis_idxs, 2],
               c="darkblue", marker='o', s=1)
    ax.scatter(X_hit[:, 1], X_hit[:, 2],
               c="darkred", marker='o', s=1)
    ax.set_title('Map LiDAR hits and misses')
    ax.set_xlim((q_minmax[0], q_minmax[1]))
    ax.set_ylim((q_minmax[2], q_minmax[3]))

    ax = plt.subplot(1, 5, 2)
    ax.scatter(X_hit[:, 1], X_hit[:, 2], c="darkred", marker='o', s=3)
    scattr = ax.scatter(qhinge_mean[:, 0], qhinge_mean[:, 1],
                        c=qgamma_mean_forward_plot.flatten(),
                        marker='o', cmap="rainbow",
                        s=10 * (1 / qgamma_mean_forward_plot.flatten()),
                        alpha=0.7)
    cbar = plt.colorbar(scattr)
    ax.set_title('Kernel lengthscales')
    ax.set_xlim((q_minmax[0], q_minmax[1]))
    ax.set_ylim((q_minmax[2], q_minmax[3]))

    ax = plt.subplot(1, 5, 3)
    ax.scatter(X_hit[:, 1], X_hit[:, 2], c="darkred", marker='o', s=3)
    scattr = ax.scatter(qhinge_mean[:, 0], qhinge_mean[:, 1],
                        c=qw_mean.flatten(),
                        marker='o', cmap="rainbow",
                        s=(qw_mean.flatten() + 0.1 - qw_mean.min()), alpha=0.7)
    cbar = plt.colorbar(scattr)
    ax.set_title('Weight mean')
    ax.set_xlim((q_minmax[0], q_minmax[1]))
    ax.set_ylim((q_minmax[2], q_minmax[3]))

    ax = plt.subplot(1, 5, 4)
    ax.scatter(X_hit[:, 1], X_hit[:, 2], c="darkred", marker='o', s=3)
    scattr = ax.scatter(qhinge_mean[:, 0], qhinge_mean[:, 1],
                        c=qw_var.flatten(),
                        marker='o', cmap="rainbow",
                        s=np.clip(200 * qw_var.flatten() / qw_var.max(), 4,
                                  np.inf), alpha=0.7)
    cbar = plt.colorbar(scattr)
    ax.set_title('Weight variance')
    ax.set_xlim((q_minmax[0], q_minmax[1]))
    ax.set_ylim((q_minmax[2], q_minmax[3]))

    ax = plt.subplot(1, 5, 5)
    contourf_res = 100
    probs_chunk1, qmesh_x1_chunk1, qmesh_x2_chunk1 = evaluate_model_probs(
        qhinge_grid_mean=qhinge_mean,
        qgamma_mean_forward=qgamma_mean_forward,
        qw_mean=data["qw_mean"],
        qw_var=data["qw_var"],
        qcellres=(0.25, 0.25),
        qminmax=q_minmax)
    contf = ax.contourf(qmesh_x1_chunk1, qmesh_x2_chunk1,
                        probs_chunk1.reshape(qmesh_x1_chunk1.shape),
                        contourf_res, cmap='jet', vmin=0, vmax=1)
    occupancy_ticks = np.linspace(0.0, 1.0, 5)
    ax.set_title("Predicted Occupancy")
    cbar = plt.colorbar(contf, ticks=occupancy_ticks)

    THE_SECOND_FIGURE = plt.figure(dpi=200, figsize=(4, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Lidar scans for 2D train chunks")

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Corresponding 2D chunk lengthscales")

    for i in range(len(train_chunks)):
        plot_offset = np.array(
            np.divmod(i, int(np.sqrt(4)))) * 1.5 * lidar_radius

        chunk = train_chunks[i]
        if chunk.N_hit > 0:
            ax1.scatter(chunk.X_miss[:, 0] + plot_offset[0],
                        chunk.X_miss[:, 1] + plot_offset[1],
                        c="darkblue", marker='o', s=1)
            ax1.scatter(chunk.X_hit[:, 0] + plot_offset[0],
                        chunk.X_hit[:, 1] + plot_offset[1],
                        c="darkred", marker='o', s=3)

        else:
            ax1.scatter(chunk.X_miss[:, 0] + plot_offset[0],
                        chunk.X_miss[:, 1] + plot_offset[1],
                        c="darkblue", marker='o', s=1)
        scattr = ax2.scatter(chunk.qhinge_grid_mean[:, 0] + plot_offset[0],
                             chunk.qhinge_grid_mean[:, 1] + plot_offset[1],
                             c=chunk.qgamma_mean_forward,
                             marker='o', cmap="rainbow",
                             s=4 * 1 / chunk.qgamma_mean_forward + 12,
                             alpha=0.7)

    plt.tight_layout()
    plt.show()
