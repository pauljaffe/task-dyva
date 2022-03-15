import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr

from task_dyva.utils import save_figure
from task_dyva.visualization import BarPlot, PlotModelLatents
from task_dyva.model_analysis import LatentSeparation


class Figure4():
    """Analysis methods and plotting routines to reproduce
    Figure 4 from the manuscript (dynamical origins of the switch cost).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    A_id = 438
    D_id = 195
    figsize = (9, 5.5)
    figdpi = 300
    palette = 'viridis'

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.A_stats = {'mv': 2, 'pt': 2, 'cue': 1}
        self.B_stats = []
        self.num_low_N = [] # Keep track of exclusions for panel B
        self.num_constant = []
        self.C_dists = []
        self.C_switch_costs = []
        self.E_stats = {'sc_plus_centroid_d': [], 'sc_minus_centroid_d': []}

    def make_figure(self):
        print('Making Figure 4...')
        print('Note: PearsonRConstantInputWarning is handled sensibly; see Methods')
        self._run_preprocessing()
        print('Stats for Figure 4')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'Fig4')
        print('')

    def _run_preprocessing(self):
        for expt_str, uid, sc in zip(self.expts, 
                                     self.user_ids, 
                                     self.sc_status):

            # Load stats from the holdout data
            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                expt_stats = pickle.load(path)

            # Distance to task centroids, stay vs. switch trials
            latent_sep = LatentSeparation(expt_stats)
            dist_stats = latent_sep.analyze()
            if sc != 'sc-':
                self.B_stats.append(dist_stats['mean_r'])
                self.num_low_N.append(dist_stats['num_low_N'])
                self.num_constant.append(dist_stats['num_constant'])
                self.C_switch_costs.append(expt_stats.summary_stats['m_switch_cost'])
                self.C_dists.append(dist_stats['normed_centroid_dist'])

            # Comparison of sc- vs. sc+ models
            if sc in ['sc+', 'sc-']:
                # Task centroid separation
                if sc == 'sc+':
                    self.E_stats['sc_plus_centroid_d'].append(
                        dist_stats['normed_centroid_dist'])
                elif sc == 'sc-':
                    self.E_stats['sc_minus_centroid_d'].append(
                        dist_stats['normed_centroid_dist'])

            # Examples
            if uid == self.A_id and sc != 'sc-':
                self.A_stats['stats'] = expt_stats
            if uid == self.D_id and sc == 'sc+':
                self.D_sc_plus_stats = expt_stats
            if uid == self.D_id and sc == 'sc-':
                self.D_sc_minus_stats = expt_stats

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(9, 14)

        # Panel A: Example model latent state trajectories
        axA = fig.add_subplot(gs[0:3, 0:3], projection='3d')
        self._make_ex_panel(axA, self.A_stats)

        # Panel B: Histogram of corr. coeffs.
        axB = fig.add_subplot(gs[0:2, 4:6])
        self._make_panel_B(axB)

        # Panel C: Across model scatter
        axC = fig.add_subplot(gs[0:2, 7:9])
        self._make_panel_C(axC)

        # Panel D: Example sc+ vs. sc- latent state trajectories
        axD1 = fig.add_subplot(gs[3:7, 0:4], projection='3d')
        axD2 = fig.add_subplot(gs[3:7, 4:8], projection='3d')
        self._make_panel_D(axD1, axD2)

        # Panel E: Distance to task centroid, sc+ vs. sc- models
        axE = fig.add_subplot(gs[3:6, 9:10])
        self._make_panel_E(axE)

        return fig

    def _make_ex_panel(self, ax, params):
        t_post = 1000
        elev, azim = 30, 60
        stats = params.pop('stats')
        plotter = PlotModelLatents(stats, post_on_dur=t_post,
                                   plot_pre_onset=False)
        _ = plotter.plot_stay_switch(ax, params, elev=elev, azim=azim)
        return ax

    def _make_panel_B(self, ax):
        print('Panel B, stats on exclusions')
        np_low_N = np.array(self.num_low_N)
        models_with_low_N = np.count_nonzero(np_low_N)
        mean_num_low_N = np.mean(np_low_N[np.nonzero(np_low_N)])
        sem_num_low_N = np.std(np_low_N[np.nonzero(np_low_N)]) / np.sqrt(models_with_low_N)
        np_constant = np.array(self.num_constant)
        models_with_constant = np.count_nonzero(np_constant)
        mean_num_constant = np.mean(np_constant[np.nonzero(np_constant)])
        print('N models with stimulus combinations excluded from within model ' \
              f'correlation summary due to low N: {models_with_low_N}; ' \
              'mean +/- s.e.m. exclusions within those models: ' \
              f'{mean_num_low_N} +/- {sem_num_low_N}')
        print('N models with stimulus combinations excluded from within model ' \
              f'correlation summary due to constant RT: {models_with_constant}; ' \
              f'mean exclusions within those models: {mean_num_constant}')
        print('-----------------------------')

        print('Panel B, main summary stats')
        N = len(self.B_stats)
        _, p = wilcoxon(self.B_stats, mode='approx')
        n_pos = np.count_nonzero(np.nonzero(np.array(self.B_stats) > 0))
        print(f'Mean +/- s.e.m. corr. within model, dist vs. model RTs: ' \
              f'{np.mean(self.B_stats)} +/- {np.std(self.B_stats) / np.sqrt(N)}')
        print(f'p = {p}')
        print(f'Num. models with positive corr.: {n_pos}')
        print(f'Fraction of models with positive corr.: {n_pos / N}')
        cmap = sns.color_palette(self.palette, as_cmap=True)
        hist_color = (0.2719, 0.6549, 0.4705, 1.0)
        mean_color = 'k'
        sns.histplot(self.B_stats, bins=np.arange(-1, 1.05, 0.05), ax=ax, color=hist_color)
        ax.scatter(np.mean(self.B_stats), 1, s=6, color=mean_color, marker='v', zorder=2)
        ax.set_xlabel("Mean Pearsons's r between distance\nto task centroid and RT (switch trials)")
        ax.set_ylabel('Count')
        ax.set_xlim([-0.75, 0.75])
        ax.set_ylim([0, 20])
        ax.set_yticks([0, 5, 10, 15, 20])
        print('-----------------------------')

    def _make_panel_C(self, ax, line_ext=10):
        ds = self.C_dists
        scs = self.C_switch_costs

        # Plot best fit line
        plot_x = np.array([min(scs) - line_ext, 
                           max(scs) + line_ext])
        m, b = np.polyfit(scs, ds, 1)
        ax.plot(plot_x, m * plot_x + b, 'k-', zorder=2, linewidth=0.5)
        
        # Plot all models
        ax.scatter(scs, ds, s=0.5, marker='o', zorder=1)
        ax.set_xlabel('Model switch cost')
        ax.set_ylabel('Normalized distance between\ntask centroids (a.u.)')

        # Stats
        r, p = pearsonr(ds, scs)
        print(f'Panel C, normed dist. vs. model switch costs: r = {r}, p = {p}')
        print('-----------------------------')

    def _make_panel_D(self, ax1, ax2):
        t_post = 1200
        elev, azim = 30, 60
        # Same axis limits for both plots
        p1_kwargs = {'xlim': [-20, 20], 'ylim': [-10, 8], 'zlim': [-10, 10]}
        p2_kwargs = p1_kwargs.copy()
        p2_kwargs['annotate'] = False

        # Plot
        plotter1 = PlotModelLatents(self.D_sc_plus_stats, post_on_dur=t_post,
                                    plot_pre_onset=False)
        _ = plotter1.plot_main_conditions(ax1, elev=elev, azim=azim, 
                                          plot_task_centroid=True, **p1_kwargs)
        plotter2 = PlotModelLatents(self.D_sc_minus_stats, post_on_dur=t_post,
                                    plot_pre_onset=False)
        _ = plotter2.plot_main_conditions(ax2, elev=elev, azim=azim,
                                          plot_task_centroid=True, **p2_kwargs)

    def _make_panel_E(self, ax):
        df = pd.DataFrame(self.E_stats)
        keys = ['sc_plus_centroid_d', 'sc_minus_centroid_d']
        labels = ['sc+ models', 'sc- models']
        ylabel = 'Normalized distance between\ntask centroids (a.u.)'
        xlim = [-0.75, 1.75]
        ylim = [0, 0.75]
        error_type = 'sem'
        kwargs = {'xticklabels': labels, 'ylabel': ylabel,
                  'xlim': xlim, 'ylim': ylim}
        barp = BarPlot(df)
        _ = barp.plot_bar(keys, error_type, ax, **kwargs)
        
        # Stats
        print('Panel E, sc+ vs. sc- distance stats:')
        w, p = wilcoxon(df['sc_plus_centroid_d'].values, 
                        y=df['sc_minus_centroid_d'].values,
                        mode='approx')
        print('sc+ vs. sc- distance b/w task centroids, signed-rank test: ' \
              f'w = {w}, p = {p}, N = {len(df)}')
        print('----------------------------------------')
