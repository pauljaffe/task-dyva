import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from task_dyva.utils import save_figure, pearson_bootstrap
from task_dyva.visualization import PlotModelLatents, BarPlot


class FigureS7():
    """Analysis methods and plotting routines to reproduce
    Figure S7 from the manuscript (dynamical origins of the switch cost;
    example trajectories; additional analysis related to Fig. 4). 
    """

    analysis_dir = 'model_analysis'
    outputs_fn = 'holdout_outputs_01SD.pkl'
    summary_fn = 'summary.pkl'
    noise_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
    noise_sds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    metrics = ['m_switch_cost', 'normed_centroid_dist']
    age_bin_strs = ['ages20to29', 'ages30to39', 'ages40to49', 'ages50to59',
                    'ages60to69', 'ages70to79', 'ages80to89']
    age_bin_labels = ['20-29', '30-39', '40-49', '50-59', 
                      '60-69', '70-79', '80-89']
    figsize = (10, 2.5)
    figdpi = 300
    palette = 'viridis'

    def __init__(self, model_dir, save_dir, metadata, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05

        # Containers for summary stats
        self.ex = {4913: {'mv': 0, 'pt': 0, 'cue': 1},
                   3538: {'mv': 1, 'pt': 1, 'cue': 0},
                   3531: {'mv': 2, 'pt': 2, 'cue': 1},
                   1276: {'mv': 2, 'pt': 2, 'cue': 0}}

        stat_dict = {k: [] for k in self.metrics}
        self.group_stats = {n: copy.deepcopy(stat_dict)
                            for n in self.noise_labels}
        self.age_data = []

    def make_figure(self):
        print('Making Figure S7...')
        self._run_preprocessing()
        print('------------------')
        fig = self._plot_figure()
        save_figure(fig, self.save_dir, 'FigS7')
        print('')

    def _run_preprocessing(self):
        for expt_str, ab, uid, sc in zip(self.expts, 
                                         self.age_bins,
                                         self.user_ids, 
                                         self.sc_status):

            if sc == 'sc-':
                continue

            # Panel d (different noise levels)
            summary_path = os.path.join(self.model_dir, expt_str, 
                                        self.analysis_dir, self.summary_fn)
            with open(summary_path, 'rb') as path:
                this_summary = pickle.load(path)
            for n in self.noise_labels:
                for key in self.metrics:
                    self.group_stats[n][key].append(this_summary[n][key])

            # Panel e (age)
            self.age_data.append([ab,
                this_summary['01']['normed_centroid_dist']])

            # Examples
            if uid in self.ex.keys():
                # Load stats from the holdout data
                outputs_path = os.path.join(self.model_dir, expt_str, 
                                            self.analysis_dir, self.outputs_fn)
                with open(outputs_path, 'rb') as path:
                    outputs = pickle.load(path)
                self.ex[uid]['stats'] = outputs

        self.age_df = pd.DataFrame(self.age_data, 
            columns=['age_bin', 'normed_centroid_dist'])

    def _plot_figure(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(7, 16)

        # Panel C: Example model latent state trajectories
        for i, uid in enumerate(self.ex.keys()):
            ax = fig.add_subplot(gs[0:3, i*4:i*4+3], projection='3d')
            params = self.ex[uid]
            _ = self._make_ex_panel(ax, params)

        # Panel D: Centroid distance vs. switch cost (different noise levels)
        axD = fig.add_subplot(gs[4:, 0:3])
        self._make_panel_D(axD)

        # Panel E: Centroid distance vs. age
        axE = fig.add_subplot(gs[4:, 4:6])
        self._make_panel_E(axE)

        return fig

    def _make_ex_panel(self, ax, params):
        t_post = 1000
        elev, azim = 30, 45
        stats = params.pop('stats')
        plotter = PlotModelLatents(stats, post_on_dur=t_post,
                                   plot_pre_onset=False)
        _ = plotter.plot_stay_switch(ax, params, elev=elev, azim=azim)
        return ax

    def _make_panel_D(self, ax):
        print('Stats for panel D:')
        r_vals = []
        cis = np.zeros((2, len(self.noise_labels)))
        for ind, n, sd in zip(range(len(self.noise_labels)), self.noise_labels, 
                              self.noise_sds):
            switch_costs = self.group_stats[n]['m_switch_cost']
            dists = self.group_stats[n]['normed_centroid_dist']
            r, p, ci_lo, ci_hi = pearson_bootstrap(switch_costs, dists, self.rng,
                                                   n_boot=self.n_boot, 
                                                   alpha=self.alpha)
            r_vals.append(r)
            cis[0, ind] = r - ci_lo
            cis[1, ind] = ci_hi - r
            print(f"Pearson's r at noise = {sd}SD: r = {r}, p = {p}")

        ax.errorbar(self.noise_sds, r_vals, yerr=cis, linestyle='-',
                    linewidth=0.5)
        ax.set_xlabel('Noise SD')
        ax.set_ylabel("Task centroid vs. switch cost\nPearson's r")
        ax.set_xticks(self.noise_sds)
        ax.set_xticklabels(self.noise_sds)
        print('-------------------------------')

    def _make_panel_E(self, ax):
        error_type = 'sem'
        xlabel = 'Age bin (years)'
        ylabel = 'Normalized distance between\ntask centroids (a.u.)'
        params = {'xticklabels': self.age_bin_labels,
                  'xlabel': xlabel,
                  'ylabel': ylabel,
                  'width': 0.65}
        bar = BarPlot(self.age_df)
        _, data = bar.alt_plot_bar('age_bin', self.age_bin_strs,
                                   'normed_centroid_dist', error_type, ax, **params)

        # Stats
        print(f'Stats for panel E: ')
        f_stat, p = f_oneway(*data)
        print(f'One-way ANOVA F stat = {f_stat}, p = {p}')
        tukey = pairwise_tukeyhsd(endog=self.age_df['normed_centroid_dist'],
                                  groups=self.age_df['age_bin'],
                                  alpha=0.05)
        print(tukey.summary())
        print('p-values:')
        print(tukey.pvalues)
