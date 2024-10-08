import os
import pickle
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, PearsonRConstantInputWarning

from task_dyva.utils import save_figure, pearson_bootstrap, adjust_boxplot
from task_dyva.visualization import PlotModelLatents
from task_dyva.model_analysis import LatentSeparation


class Figure4:
    """Analysis methods and plotting routines to reproduce
    Figure 4 from the manuscript (dynamical origins of the switch cost).
    """

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"
    A_id = 438
    D_id = 195
    figsize = (9, 5.5)
    figdpi = 300
    palette = "viridis"
    # box_colors = {'Participants': (0.2363, 0.3986, 0.5104, 1.0),
    #              'Models': (0.2719, 0.6549, 0.4705, 1.0)}
    box_colors = [(0.2363, 0.3986, 0.5104, 1.0), (0.2719, 0.6549, 0.4705, 1.0)]

    def __init__(self, model_dir, save_dir, metadata, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata["name"]
        self.user_ids = metadata["user_id"]
        self.sc_status = metadata["switch_cost_type"]
        self.exgauss = metadata["exgauss"]
        self.early = metadata["early"]
        self.optimal = metadata["optimal"]
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05
        # Constant input warning refers to calculations not included in summary analyses
        warnings.simplefilter("ignore", PearsonRConstantInputWarning)

        # Containers for summary stats
        self.A_stats = {"mv": 2, "pt": 2, "cue": 1}
        self.B_stats = []
        self.num_low_N = []  # Keep track of exclusions for panel B
        self.num_constant = []
        self.C_dists = []
        self.C_switch_costs = []
        self.E_stats = {"sc_plus_centroid_d": [], "sc_minus_centroid_d": []}
        self.E_plot = {"expt": [], "model_type": [], "value": []}

    def make_figure(self):
        print("Making Figure 4...")
        self._run_preprocessing()
        print("Stats for Figure 4")
        print("------------------")
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, "Fig4")
        print("")

    def _run_preprocessing(self):
        for expt_str, uid, sc, exg, early, opt in zip(
            self.expts,
            self.user_ids,
            self.sc_status,
            self.exgauss,
            self.early,
            self.optimal,
        ):
            if exg == "exgauss+" or early or opt:
                continue

            # Load stats from the holdout data
            stats_path = os.path.join(
                self.model_dir, expt_str, self.analysis_dir, self.stats_fn
            )
            with open(stats_path, "rb") as path:
                expt_stats = pickle.load(path)

            # Distance to task centroids, stay vs. switch trials
            latent_sep = LatentSeparation(expt_stats)
            dist_stats = latent_sep.analyze()
            if sc != "sc-":
                self.B_stats.append(dist_stats["mean_r"])
                self.num_low_N.append(dist_stats["num_low_N"])
                self.num_constant.append(dist_stats["num_constant"])
                self.C_switch_costs.append(
                    expt_stats.summary_stats["m_switch_cost"]
                )
                self.C_dists.append(dist_stats["normed_centroid_dist"])

            # Comparison of sc- vs. sc+ models
            if sc in ["sc+", "sc-"]:
                # Task centroid separation
                self.E_plot["expt"].append(expt_str)
                self.E_plot["value"].append(dist_stats["normed_centroid_dist"])
                if sc == "sc+":
                    self.E_stats["sc_plus_centroid_d"].append(
                        dist_stats["normed_centroid_dist"]
                    )
                    self.E_plot["model_type"].append("sc+ models")
                elif sc == "sc-":
                    self.E_stats["sc_minus_centroid_d"].append(
                        dist_stats["normed_centroid_dist"]
                    )
                    self.E_plot["model_type"].append("sc- models")

            # Examples
            if uid == self.A_id and sc != "sc-":
                self.A_stats["stats"] = expt_stats
            if uid == self.D_id and sc == "sc+":
                self.D_sc_plus_stats = expt_stats
            if uid == self.D_id and sc == "sc-":
                self.D_sc_minus_stats = expt_stats

    def _plot_figure_get_stats(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(9, 14)

        # Panel A: Example model latent state trajectories
        axA = fig.add_subplot(gs[0:3, 0:3], projection="3d")
        self._make_ex_panel(axA, self.A_stats)

        # Panel B: Histogram of corr. coeffs.
        axB = fig.add_subplot(gs[0:2, 4:6])
        self._make_panel_B(axB)

        # Panel C: Across model scatter
        axC = fig.add_subplot(gs[0:2, 7:9])
        self._make_panel_C(axC)

        # Panel D: Example sc+ vs. sc- latent state trajectories
        axD1 = fig.add_subplot(gs[3:7, 0:4], projection="3d")
        axD2 = fig.add_subplot(gs[3:7, 4:8], projection="3d")
        self._make_panel_D(axD1, axD2)

        # Panel E: Distance to task centroid, sc+ vs. sc- models
        axE = fig.add_subplot(gs[3:6, 9:10])
        self._make_panel_E(axE)

        return fig

    def _make_ex_panel(self, ax, params):
        t_post = 1000
        elev, azim = 30, 60
        stats = params.pop("stats")
        plotter = PlotModelLatents(
            stats, post_on_dur=t_post, plot_pre_onset=False
        )
        _ = plotter.plot_stay_switch(ax, params, elev=elev, azim=azim)
        return ax

    def _make_panel_B(self, ax):
        print("Panel B, stats on exclusions")
        np_low_N = np.array(self.num_low_N)
        models_with_low_N = np.count_nonzero(np_low_N)
        mean_num_low_N = np.mean(np_low_N[np.nonzero(np_low_N)])
        sem_num_low_N = np.std(np_low_N[np.nonzero(np_low_N)]) / np.sqrt(
            models_with_low_N
        )
        print(
            "N models with stimulus combinations excluded from within model "
            f"correlation summary due to low N: {models_with_low_N}; "
            "mean +/- s.e.m. exclusions within those models: "
            f"{mean_num_low_N} +/- {sem_num_low_N}"
        )
        print("-----------------------------")

        print("Panel B, main summary stats")
        N = len(self.B_stats)
        w, p = wilcoxon(self.B_stats, mode="approx")
        n_pos = np.count_nonzero(np.nonzero(np.array(self.B_stats) > 0))
        print(
            f"Mean +/- s.e.m. corr. within model, dist vs. model RTs: "
            f"{np.mean(self.B_stats)} +/- {np.std(self.B_stats) / np.sqrt(N)}"
        )
        print(f"Signed-rank test: p = {p}, W = {w}")
        print(f"Total models analyzed: {N}")
        print(f"Num. models with positive corr.: {n_pos}")
        print(f"Fraction of models with positive corr.: {n_pos / N}")
        hist_color = (0.2719, 0.6549, 0.4705, 1.0)
        mean_color = "k"
        sns.histplot(
            self.B_stats,
            bins=np.arange(-1, 1.05, 0.05),
            ax=ax,
            color=hist_color,
        )
        ax.scatter(
            np.mean(self.B_stats),
            1,
            s=6,
            color=mean_color,
            marker="v",
            zorder=2,
        )
        ax.set_xlabel(
            "Mean Pearsons's r between distance\nto task centroid and RT (switch trials)"
        )
        ax.set_ylabel("Count")
        ax.set_xlim([-0.75, 0.75])
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        print("-----------------------------")

    def _make_panel_C(self, ax, line_ext=10):
        ds = self.C_dists
        scs = self.C_switch_costs

        # Plot best fit line
        plot_x = np.array([min(scs) - line_ext, max(scs) + line_ext])
        m, b = np.polyfit(scs, ds, 1)
        ax.plot(plot_x, m * plot_x + b, "k-", zorder=2, linewidth=0.5)

        # Plot all models
        ax.scatter(scs, ds, s=0.5, marker="o", zorder=1)
        ax.set_xlabel("Model switch cost (ms)")
        ax.set_ylabel("Normalized distance between\ntask centroids (a.u.)")

        # Stats
        print("Panel C stats: normed dist. vs. model switch costs")
        r, p, ci_lo, ci_hi = pearson_bootstrap(
            scs, ds, self.rng, n_boot=self.n_boot, alpha=self.alpha
        )
        p_str = "{:0.2e}".format(p)
        tstr = f"r = {round(r, 2)}, 95% CI: ({round(ci_lo, 2)}, {round(ci_hi, 2)}), p = {p_str}"
        print(tstr)
        print(f"Best-fit slope: {m}; intercept: {b}")
        mean_d = np.mean(ds)
        sem_d = np.std(ds) / np.sqrt(len(ds))
        print(f"Mean +/- s.e.m. centroid distance: {mean_d} +/- {sem_d}")
        mean_sc = np.mean(scs)
        sem_sc = np.std(scs) / np.sqrt(len(scs))
        print(f"Mean +/- s.e.m. switch cost (ms): {mean_sc} +/- {sem_sc}")
        print("-----------------------------")

    def _make_panel_D(self, ax1, ax2):
        t_post = 1200
        elev, azim = 30, 60
        # Same axis limits for both plots
        p1_kwargs = {"xlim": [-20, 20], "ylim": [-10, 8], "zlim": [-10, 10]}
        p2_kwargs = p1_kwargs.copy()
        p2_kwargs["annotate"] = False

        # Plot
        plotter1 = PlotModelLatents(
            self.D_sc_plus_stats, post_on_dur=t_post, plot_pre_onset=False
        )
        _ = plotter1.plot_main_conditions(
            ax1, elev=elev, azim=azim, plot_task_centroid=True, **p1_kwargs
        )
        plotter2 = PlotModelLatents(
            self.D_sc_minus_stats, post_on_dur=t_post, plot_pre_onset=False
        )
        _ = plotter2.plot_main_conditions(
            ax2, elev=elev, azim=azim, plot_task_centroid=True, **p2_kwargs
        )

    def _make_panel_E(self, ax):
        df = pd.DataFrame(self.E_stats)
        df_plot = pd.DataFrame(self.E_plot)
        labels = ["sc+ models", "sc- models"]
        ylabel = "Normalized distance between\ntask centroids (a.u.)"
        sns.boxplot(
            data=df_plot,
            x="model_type",
            y="value",
            ax=ax,
            orient="v",
            fliersize=1,
            linewidth=0.5,
            palette=self.box_colors,
        )
        box_params = {
            "ylabel": ylabel,
            "ylim": [0, 1],
            "xlim": [-1, 2],
            "xticklabels": labels,
            "plot_legend": False,
        }
        adjust_boxplot(ax, **box_params)

        # Stats
        print("Panel E, sc+ vs. sc- distance stats:")
        w, p = wilcoxon(
            df["sc_plus_centroid_d"].values,
            y=df["sc_minus_centroid_d"].values,
            mode="approx",
        )
        print(
            "sc+ vs. sc- distance b/w task centroids, signed-rank test: "
            f"W = {w}, p = {p}, N = {len(df)}"
        )
        print("----------------------------------------")
