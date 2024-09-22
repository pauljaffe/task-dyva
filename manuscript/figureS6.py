import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva import Experiment
from task_dyva.utils import save_figure, plot_scatter
from task_dyva.visualization import BarPlot


class FigureS6:
    """Analysis for Figure S6: models trained with
    optimal training protocol. Example in panel A
    and summary stats (B-F) are saved in separate
    figures.
    """

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"
    line_ext = 10
    stats_figsize = (6.5, 1.5)
    figdpi = 300
    palette = "viridis"
    ex_str = "optimal_square2"
    ex_noise_params = {"noise_type": "indep", "noise_sd": 0.1}
    ex_kwargs = {
        "logger_type": None,
        "test": ex_noise_params,
        "mode": "testing",
        "params_to_load": "model_params.pth",
    }
    ex_raw_fn = "data_pre_split.pickle"
    ex_ind = 18
    device = "cpu"

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.metadata = metadata
        self.expts = metadata["name"]
        self.user_ids = metadata["user_id"]
        self.exgauss = metadata["exgauss"]
        self.early = metadata["early"]
        self.sc_status = metadata["switch_cost_type"]
        self.optimal = metadata["optimal"]
        fontsize = 6
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["axes.labelsize"] = fontsize
        plt.rcParams["axes.titlesize"] = fontsize
        plt.rcParams["xtick.labelsize"] = fontsize
        plt.rcParams["ytick.labelsize"] = fontsize
        plt.rcParams["legend.fontsize"] = fontsize

        # Containers for summary stats
        self.stats = {
            "m_switch_cost": [],
            "m_mean_rt": [],
            "m_con_effect": [],
            "m_accuracy": [],
            "m_rt_sd": [],
        }

    def make_figure(self):
        print("Making Figure S6...")
        self._run_preprocessing()
        print("Stats for Figure S6")
        print("------------------")
        ex_fig, stats_fig = self._plot_figure_get_stats()
        save_figure(ex_fig, self.save_dir, "FigS6A")
        save_figure(stats_fig, self.save_dir, "FigS6B-F")
        print("")

    def _run_preprocessing(self):
        for expt_str, uid, exg, earl, sc, opt in zip(
            self.expts,
            self.user_ids,
            self.exgauss,
            self.early,
            self.sc_status,
            self.optimal,
        ):
            if not opt:
                continue

            stats_path = os.path.join(
                self.model_dir, expt_str, self.analysis_dir, self.stats_fn
            )
            with open(stats_path, "rb") as path:
                expt_stats = pickle.load(path)
            for key in self.stats.keys():
                self.stats[key].append(expt_stats.summary_stats[key])
            if expt_str == self.ex_str:
                ex_dir = os.path.join(self.model_dir, expt_str)
                expt = Experiment(
                    ex_dir,
                    ex_dir,
                    self.ex_raw_fn,
                    expt_str,
                    processed_dir=ex_dir,
                    device=self.device,
                    **self.ex_kwargs,
                )
                self.ex_expt = expt

    def _plot_figure_get_stats(self):
        print("Models trained with optimal training protocol:")
        # Panel a: example output
        ex_fig = self._make_panel_A()

        # Panel b: accuracy
        stats_fig = plt.figure(
            constrained_layout=False,
            figsize=self.stats_figsize,
            dpi=self.figdpi,
        )
        gs = stats_fig.add_gridspec(8, 27)
        B_ax = stats_fig.add_subplot(gs[:, :3])
        self._stats_panel(B_ax, "m_accuracy", "Accuracy")

        # Panel c: mean RT
        C_ax = stats_fig.add_subplot(gs[:, 6:9])
        self._stats_panel(C_ax, "m_mean_rt", "Model mean RT (ms)")

        # Panel d: switch cost
        D_ax = stats_fig.add_subplot(gs[:, 12:15])
        self._stats_panel(D_ax, "m_switch_cost", "Model switch cost (ms)")

        # panel e: congruency effect
        E_ax = stats_fig.add_subplot(gs[:, 18:21])
        self._stats_panel(E_ax, "m_con_effect", "Model congruency effect (ms)")

        # panel f: RT std.dev.
        F_ax = stats_fig.add_subplot(gs[:, 24:])
        self._stats_panel(F_ax, "m_rt_sd", "Model RT SD (ms)")

        return ex_fig, stats_fig

    def _make_panel_A(self):
        ex_fig = self.ex_expt.plot_generated_sample(
            "NA",
            self.ex_expt.test_dataset,
            sample_ind=self.ex_ind,
            for_paper=True,
        )
        return ex_fig[1]

    def _stats_panel(self, ax, stat, label):
        d = self.stats[stat]
        m = np.mean(d)
        sem = np.std(d) / np.sqrt(len(d))
        self._plot_mean_sem(0, m, sem, ax)
        sns.swarmplot(y=d, orient="v", size=2, ax=ax, alpha=0.8)
        ax.set_ylabel(label)
        ax.set_xlim([-0.3, 0.3])
        w, p = wilcoxon(d, mode="exact")
        print(f"{stat} mean +/- s.e.m.: {m} +/- {sem}")
        print(f"Signed-rank test vs. zero: w = {w}, p = {p}")
        print("-----")

    def _plot_mean_sem(self, x, mean, sem, ax):
        ax.plot(x, mean, "k_", markerfacecolor="k", ms=5, mew=1)
        ax.plot([x, x], [mean - sem, mean + sem], "k-", linewidth=1)
