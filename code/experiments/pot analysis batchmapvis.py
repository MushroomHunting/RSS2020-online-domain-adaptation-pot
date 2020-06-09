import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sbn
from time import time
from hotpot.transfer_datasets import load_learned_data
from hotpot.hilbert_maps import evaluate_model_probs_batched, eval_probs_only
from hotpot.kernels import rbf_kernel
from hotpot.util import calc_scores
import torch
from hotpot.bhm_torch_cpu import BHM2D_PYTORCH

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.6)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

np.random.seed(1111)

if __name__ == "__main__":
    # Change this to pick the domain to domain transfer
    from_to = source_domain, target_domain = (1, 2)
    do_metrics_eval = True
    do_map_vis = True
    do_full_map = False
    do_transport_refine = True
    print("source domain: {}\ntarget domain: {}".format(source_domain,
                                                        target_domain))
    transfer_domain_dict = {(1, 1): "carla2dtown1town1_full_auto_SPARSE3a.npz",
                            (1, 2): "carla2dtown1town2_full_auto_SPARSE3a.npz",
                            (1, 3): "carla2dtown1town3_full_auto_SPARSE3a.npz",
                            (2, 1): "carla2dtown2town1_full_auto_SPARSE3a.npz",
                            (2, 2): "carla2dtown2town2_full_auto_SPARSE3a.npz",
                            (2, 3): "carla2dtown2town3_full_auto_SPARSE3a.npz",
                            (3, 1): "carla2dtown3town1_full_auto_SPARSE3a.npz",
                            (3, 2): "carla2dtown3town2_full_auto_SPARSE3a.npz",
                            (3,
                             3): "carla2dtown3town3_full_auto_SPARSE3a.npz", }

    if do_metrics_eval is True:
        if target_domain == 1:
            data_test = np.load("carla2dtown1fullauto_test2k.npz")
        elif target_domain == 2:
            data_test = np.load("carla2dtown2fullauto_test2k.npz")
        elif target_domain == 3:
            data_test = np.load("carla2dtown3fullauto_test2k.npz")
        X_tst = data_test["X_test"][:, 1:]  # we don't need time
        y_tst = data_test["Y_test"]

    else:
        if target_domain == 1:
            data_test = load_learned_data(dataset="carla_2dtown1_full_auto")
        elif target_domain == 2:
            data_test = load_learned_data(dataset="carla_2dtown2_full_auto")
        elif target_domain == 3:
            data_test = load_learned_data(dataset="carla_2dtown3_full_auto")
        X_tst = data_test["X_train"][:, 1:]  # we don't need time
        y_tst = data_test["Y_train"]
    x_min, y_min = np.min(X_tst, axis=0)
    x_max, y_max = np.max(X_tst, axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    df = np.load(transfer_domain_dict[from_to])

    contourf_res = 100

    if "qhinge_mean" in df:
        transp_OOS_qhinge_mean_unscaled_FILTERED = df["qhinge_mean"]
    elif "qhinge_grid_mean" in df:
        transp_OOS_qhinge_mean_unscaled_FILTERED = df["qhinge_grid_mean"]
    qgamma_mean_forward_FILTERED = df["qgamma_mean_forward"]
    qw_mean_FILTERED = df["qw_mean"]
    qw_var_FILTERED = df["qw_var"]

    fig = plt.figure(dpi=200, figsize=(6, 3))
    ax = plt.subplot(1, 2, 1)
    scattr = ax.scatter(transp_OOS_qhinge_mean_unscaled_FILTERED[:, 0],
                        transp_OOS_qhinge_mean_unscaled_FILTERED[:, 1],
                        c=qgamma_mean_forward_FILTERED.flatten(),
                        marker='o', cmap="rainbow", s=3)
    cbar = plt.colorbar(scattr)
    ax.set_title('Kernels with LiDAR hits')
    ax = plt.subplot(1, 2, 2)
    qmean_filtered = transp_OOS_qhinge_mean_unscaled_FILTERED  # [, :]
    qgamma_filtered = qgamma_mean_forward_FILTERED  # [:, q_mask]
    qwmean_filtered = qw_mean_FILTERED  # [q_mask, :]
    qwvar_filtered = qw_var_FILTERED  # [q_mask, :]

    if do_full_map is True:
        q_minmax = (x_min, x_max,
                    y_min, y_max)
    else:
        """
        This is for RePOT seeded with the transported kernels and model weights
        Isolate sub-part (for Analytic BHM tractability)
        """
        if target_domain == 1:
            kernel_border = 10  # Extra space for kernels
        elif target_domain == 2:
            kernel_border = 10  # Extra space for kernels
        elif target_domain == 3:
            kernel_border = 10  # Extra space for kernels
            x_min = x_min
            x_max = x_max - (x_max - x_min) * 0.25
            y_min = y_min
            y_max = y_max - (y_max - y_min) * 0.25

        q_minmax = (x_min, x_max,
                    y_min, y_max)

        """
        This is only for RePOT
        """
        if target_domain == 1:
            data_trn = load_learned_data(dataset="carla_2dtown1_full_auto")
        elif target_domain == 2:
            data_trn = load_learned_data(dataset="carla_2dtown2_full_auto")
        elif target_domain == 3:
            data_trn = load_learned_data(dataset="carla_2dtown3_full_auto")
        X_trn = data_trn["X_train"][:, 1:]  # we don't need time
        y_trn = data_trn["Y_train"]
        X_trn_mask = ((X_trn[:, 0] >= x_min) &
                      (X_trn[:, 0] <= x_max) &
                      (X_trn[:, 1] >= y_min) &
                      (X_trn[:, 1] <= y_max))
        X_trn = X_trn[X_trn_mask, :]
        y_trn = y_trn[X_trn_mask]

        X_tst_mask = ((X_tst[:, 0] >= x_min) &
                      (X_tst[:, 0] <= x_max) &
                      (X_tst[:, 1] >= y_min) &
                      (X_tst[:, 1] <= y_max))
        qmean_filtered_mask = ((qmean_filtered[:, 0] >= x_min - kernel_border) &
                               (qmean_filtered[:, 0] <= x_max + kernel_border) &
                               (qmean_filtered[:, 1] >= y_min - kernel_border) &
                               (qmean_filtered[:, 1] <= y_max + kernel_border))

        X_tst = X_tst[X_tst_mask, :]
        y_tst = y_tst[X_tst_mask]

        qmean_filtered = qmean_filtered[qmean_filtered_mask, :]
        qgamma_filtered = qgamma_filtered[:, qmean_filtered_mask]
        qwmean_filtered = qwmean_filtered[qmean_filtered_mask, :]
        qwvar_filtered = qwvar_filtered[qmean_filtered_mask, :]

        """
        # Refining with BHM (RePOT), using initialised with the transported POT-HM as priors
        """
        bhm = BHM2D_PYTORCH(
            gamma=torch.tensor(qgamma_filtered.flatten(), dtype=torch.float32),
            mu=torch.tensor(qwmean_filtered.flatten(), dtype=torch.float32),
            sig=torch.tensor(qwvar_filtered.flatten(), dtype=torch.float32),
            grid=torch.tensor(qmean_filtered, dtype=torch.float32),
            nIter=5)

        X_q_features = rbf_kernel(X_tst, qmean_filtered, qgamma_filtered)

        bhm.fit(X=torch.tensor(X_q_features, dtype=torch.float32),
                y=torch.tensor(y_tst.reshape(-1, 1), dtype=torch.float32),
                X_is_feature_map=True)

        qwmean_filtered_relearned = bhm.mu.data.numpy().reshape(-1, 1)
        qwvar_filtered_relearned = bhm.sig.data.numpy().reshape(-1, 1)

    if do_metrics_eval is True:
        Y_q = eval_probs_only(X_q=X_tst,
                              qhinge_grid_mean=qmean_filtered,
                              qgamma_mean_forward=qgamma_filtered,
                              qw_mean=qwmean_filtered,
                              qw_var=qwvar_filtered)
        calc_scores('neurips19_metrics_' + transfer_domain_dict[from_to][:-4],
                    y_tst, Y_q, time_taken=-1,
                    N_points=X_tst.shape[0] / X_tst.shape[0] * 100)

        if do_transport_refine is True:
            Y_q = eval_probs_only(X_q=X_tst,
                                  qhinge_grid_mean=qmean_filtered,
                                  qgamma_mean_forward=qgamma_filtered,
                                  qw_mean=qwmean_filtered_relearned,
                                  qw_var=qwvar_filtered_relearned)
            calc_scores(
                'neurips19_refine_metrics_' + transfer_domain_dict[from_to][
                                              :-4],
                y_tst, Y_q, time_taken=-1,
                N_points=X_tst.shape[0] / X_tst.shape[0] * 100)

    if do_map_vis is True:
        tic = time()
        q_minmax = (q_minmax[0] - 10,
                    q_minmax[1] + 10,
                    q_minmax[2] - 10,
                    q_minmax[3] + 10,)
        if do_transport_refine is True:
            probs_chunk1, \
            qmesh_x1_chunk1, \
            qmesh_x2_chunk1 = evaluate_model_probs_batched(
                qhinge_grid_mean=qmean_filtered,
                qgamma_mean_forward=qgamma_filtered,
                qw_mean=qwmean_filtered_relearned,
                qw_var=qwvar_filtered_relearned,
                batch_size=500,
                qcellres=(1.0, 1.0),
                qminmax=q_minmax)
        else:
            probs_chunk1, \
            qmesh_x1_chunk1, \
            qmesh_x2_chunk1 = evaluate_model_probs_batched(
                qhinge_grid_mean=qmean_filtered,
                qgamma_mean_forward=qgamma_filtered,
                qw_mean=qwmean_filtered,
                qw_var=qwvar_filtered,
                batch_size=500,
                qcellres=(1.0, 1.0),
                qminmax=q_minmax)
        contf = ax.contourf(qmesh_x1_chunk1,
                            qmesh_x2_chunk1,
                            probs_chunk1.reshape(qmesh_x1_chunk1.shape),
                            contourf_res, cmap='jet', vmin=0, vmax=1)
        toc = time()
        print("mapping time: {}".format(np.round(toc - tic, 3)))
        occupancy_ticks = np.linspace(0.0, 1.0, 5)

        """ Overlay the lidar hits in white """
        hit_mask = (y_tst.flatten() == 1)
        X_tst_hits = X_tst[hit_mask, :]
        qminmax_mask = ((X_tst_hits[:, 0] >= q_minmax[0]) &
                        (X_tst_hits[:, 0] <= q_minmax[1]) &
                        (X_tst_hits[:, 1] >= q_minmax[2]) &
                        (X_tst_hits[:, 1] <= q_minmax[3]))
        X_tst_hits = X_tst_hits[qminmax_mask, :]

        ax = plt.subplot(1, 2, 2)
        ax.set_title("Predicted Occupancy (Atom fusion, NAIVE)")
        cbar = plt.colorbar(contf, ticks=occupancy_ticks)

    plt.tight_layout()
    plt.show()
