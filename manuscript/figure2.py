import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from task_dyva.visualization import PlotRTs, BarPlot
from task_dyva.utils import save_figure, plot_scatter, expt_stats_to_df


class Figure2():
    """Analysis methods and plotting routines to reproduce
    Figure 2 from the manuscript (behavior summary).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    line_ext = 10
    age_bin_labels = ['20-29', '30-39', '40-49', '50-59', 
                      '60-69', '70-79', '80-89']
    figsize = (7, 6.5)
    figdpi = 300
    palette = 'viridis'

    # Exemplars
    exemplar_ids = {1076: 'A1',
                    1434: 'A2',
                    2247: 'B1',
                    2347: 'B2',
                    734: 'C1',
                    910: 'C2'}

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.group_stats = {'m_switch_cost': [], 'u_switch_cost': [],
                            'm_mean_rt': [], 'u_mean_rt': [], 
                            'm_con_effect': [], 'u_con_effect': [], 
                            'u_rt_sd': [], 'm_rt_sd': []}
        self.analysis_expt_stats = []
        self.analysis_age_bins = []
        self.analysis_expt_strs = []
        self.exemplars = {}

    def make_figure(self):
        print('Making Figure 2...')
        self._run_preprocessing()
        print('Stats for Figure 2')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'Fig2')
        print('')

    def _run_preprocessing(self):
        for expt_str, ab, uid, sc in zip(self.expts, 
                                         self.age_bins, 
                                         self.user_ids, 
                                         self.sc_status):
            # Skip sc- models
            if sc == 'sc-':
                continue
            
            # Load stats from the holdout data
            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                expt_stats = pickle.load(path)
            self.analysis_age_bins.append(ab)
            self.analysis_expt_stats.append(expt_stats)
            self.analysis_expt_strs.append(expt_str)
            for key in self.group_stats.keys():
                self.group_stats[key].append(expt_stats.summary_stats[key])

            # Get stats from the example participants / models
            if uid in self.exemplar_ids.keys():
                self.exemplars[self.exemplar_ids[uid]] = expt_stats

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize,
                         dpi=self.figdpi)
        gs = fig.add_gridspec(24, 25)

        ######################################
        # Panels A-C: Example RT distributions
        ######################################
        ABC_axes = {}

        # Panel A: Mean RT
        ABC_axes['A1'] = fig.add_subplot(gs[0:6, 0:3])
        ABC_axes['A2'] = fig.add_subplot(gs[0:6, 4:7])

        # Panel B: Switch cost
        ABC_axes['B1'] = fig.add_subplot(gs[0:6, 9:12])
        ABC_axes['B2'] = fig.add_subplot(gs[0:6, 13:16])

        # Panel C: Congruency effect
        ABC_axes['C1'] = fig.add_subplot(gs[0:6, 18:21])
        ABC_axes['C2'] = fig.add_subplot(gs[0:6, 22:25])
        self._make_panels_ABC(ABC_axes)

        #################################################
        # Panels D-F: Model vs. participant scatter plots
        #################################################
        # Panel D: Mean RT
        D_params = {'ax_lims': [600, 1250],
                    'metric': 'mean_rt',
                    'label': 'mean RT (ms)'}
        D_ax = fig.add_subplot(gs[8:15, 0:7])
        plot_scatter(self.group_stats, D_params, D_ax, self.line_ext)

        # Panel E: Switch cost
        E_params = {'ax_lims': [-25, 300],
                    'metric': 'switch_cost',
                    'label': 'switch cost (ms)'}
        E_ax = fig.add_subplot(gs[8:15, 9:16])
        plot_scatter(self.group_stats, E_params, E_ax, self.line_ext)
        
        # Panel F: Congruency effect
        F_params = {'ax_lims': [0, 300],
                    'metric': 'con_effect',
                    'label': 'congruency effect (ms)'}
        F_ax = fig.add_subplot(gs[8:15, 18:25])
        plot_scatter(self.group_stats, F_params, F_ax, self.line_ext)

        #################################################
        # Panels G-I: Model vs. participant binned by age
        #################################################
        error_type = 'sem'
        metrics = ['mean_rt', 'switch_cost', 'con_effect']
        stats_df = expt_stats_to_df(metrics,
                                    self.analysis_expt_strs,
                                    self.analysis_age_bins,
                                    self.analysis_expt_stats)

        # Panel G: Mean RT
        G_df = stats_df.query("metric == 'mean_rt'")
        G_params = {'ylim': [600, 1000],
                    'ylabel': 'Mean RT (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': True}
        G_ax = fig.add_subplot(gs[17:24, 0:7])
        G_bar = BarPlot(G_df)
        _ = G_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                                   error_type, G_ax, **G_params)

        # Panel H: Switch cost
        H_df = stats_df.query("metric == 'switch_cost'")
        H_params = {'ylim': [0, 120],
                    'xlabel': 'Age bin (years)',
                    'ylabel': 'Switch cost (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': False}
        H_ax = fig.add_subplot(gs[17:24, 9:16])
        H_bar = BarPlot(H_df)
        _ = H_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                                   error_type, H_ax, **H_params)

        # Panel I: Congruency effect
        I_df = stats_df.query("metric == 'con_effect'")
        I_params = {'ylim': [0, 130],
                    'ylabel': 'Congruency effect (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': False}
        I_ax = fig.add_subplot(gs[17:24, 18:25])
        I_bar = BarPlot(I_df)
        _ = I_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                                   error_type, I_ax, **I_params)

        return fig

    def _make_panels_ABC(self, axes):
        for key in axes.keys():
            ax = axes[key]
            ex = self.exemplars[key]

            if key[0] == 'A':
                plot_type = 'all'
            elif key[0] == 'B':
                plot_type = 'switch'
            elif key[0] == 'C':
                plot_type = 'congruency'

            plotter = PlotRTs(ex)
            ax = plotter.plot_rt_dists(ax, plot_type)

            if key == 'A1':
                ax.legend(labels=['Participant', 'Model'])
                ax.get_legend().get_frame().set_linewidth(0.0)
                ax.set_ylabel('RT (ms)')
            else:
                ax.get_legend().remove()
