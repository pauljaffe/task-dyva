import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from task_dyva.visualization import BarPlot
from task_dyva.utils import save_figure, plot_scatter, expt_stats_to_df


class FigureS4():
    """Analysis methods and plotting routines to reproduce
    Figure S4 from the manuscript (RT variability).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    figsize = (9, 4.5)
    line_ext = 0.1
    age_bin_labels = ['20-29', '30-39', '40-49', '50-59', 
                      '60-69', '70-79', '80-89']
    palette = 'viridis'

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.group_stats = {'u_rt_sd': [], 'm_rt_sd': []}
        self.analysis_expt_stats = []
        self.analysis_age_bins = []
        self.analysis_expt_strs = []

    def make_figure(self):
        print('Making Figure S4...')
        self._run_preprocessing()
        print('Stats for Figure S4')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS4')
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

    def _plot_figure_get_stats(self):
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        ######################################
        # Panels A-C: Example RT distributions
        ######################################
        ABC_axes = {}

        # Panel A: Mean RT
        ABC_axes['A1'] = fig.add_subplot(gs[0:6, 0:3])
        ABC_axes['A2'] = fig.add_subplot(gs[0:6, 3:6])

        # Panel B: Switch cost
        ABC_axes['B1'] = fig.add_subplot(gs[0:6, 6:9])
        ABC_axes['B2'] = fig.add_subplot(gs[0:6, 9:12])

        # Panel C: Congruency effect
        ABC_axes['C1'] = fig.add_subplot(gs[0:6, 12:15])
        ABC_axes['C2'] = fig.add_subplot(gs[0:6, 15:18])
        self._make_panels_ABC(ABC_axes)

        #################################################
        # Panels D-F: Model vs. participant scatter plots
        #################################################

        # Panel D: Mean RT
        D_params = {'ax_lims': [600, 1250],
                    'metric': 'mean_rt',
                    'label': 'mean RT (ms)'}
        D_ax = fig.add_subplot(gs[7:13, 0:6])
        self._plot_scatter_get_stats(D_params, D_ax)

        # Panel E: Switch cost
        E_params = {'ax_lims': [-25, 300],
                    'metric': 'switch_cost',
                    'label': 'switch cost (ms)'}
        E_ax = fig.add_subplot(gs[7:13, 6:12])
        self._plot_scatter_get_stats(E_params, E_ax)
        
        # Panel F: Congruency effect
        F_params = {'ax_lims': [0, 300],
                    'metric': 'con_effect',
                    'label': 'congruency effect (ms)'}
        F_ax = fig.add_subplot(gs[7:13, 12:18])
        self._plot_scatter_get_stats(F_params, F_ax)

        #################################################
        # Panels G-I: Model vs. participant binned by age
        #################################################
        error_type = 'sem'
        stats_df = self._format_as_df()

        # Panel G: Mean RT
        G_df = stats_df.query("metric == 'mean_rt'")
        G_params = {'ylim': [600, 1000],
                    'ylabel': 'Mean RT (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': True}
        G_ax = fig.add_subplot(gs[14:20, 0:6])
        G_bar = BarPlot(G_df, self.palette)
        G_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                               error_type, G_ax, **G_params)

        # Panel H: Switch cost
        H_df = stats_df.query("metric == 'switch_cost'")
        H_params = {'ylim': [0, 120],
                    'xlabel': 'Age bin (years)',
                    'ylabel': 'Switch cost (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': False}
        H_ax = fig.add_subplot(gs[14:20, 6:12])
        H_bar = BarPlot(H_df, self.palette)
        H_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                               error_type, H_ax, **H_params)

        # Panel I: Congruency effect
        I_df = stats_df.query("metric == 'con_effect'")
        I_params = {'ylim': [0, 130],
                    'ylabel': 'Congruency effect (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': False}
        I_ax = fig.add_subplot(gs[14:20, 12:18])
        I_bar = BarPlot(I_df, self.palette)
        I_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
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
                ax.get_legend().set_title(None)
                ax.get_legend().get_frame().set_linewidth(0.0)
                ax.legend(labels=['Participant', 'Model'])
                ax.set_ylabel('RT (ms)')
            else:
                ax.get_legend().remove()

    def _plot_scatter_get_stats(self, params, ax):
        metric = params['metric']
        u_key = f'u_{metric}'
        m_key = f'm_{metric}'
        u_vals = np.array(self.group_stats[u_key])
        m_vals = np.array(self.group_stats[m_key])    
        
        # Plot best fit line
        plot_x = np.array([min(u_vals) - self.line_ext, 
                           max(u_vals) + self.line_ext])
        m, b = np.polyfit(u_vals, m_vals, 1)
        ax.plot(plot_x, m * plot_x + b, 'k-', zorder=1)
        
        # Plot all individuals
        ax.scatter(u_vals, m_vals, s=6, marker='o', zorder=2, alpha=0.75)
        ax.set_xlabel(f"Participant {params['label']}")
        ax.set_ylabel(f"Model {params['label']}")
        ax.set_xlim(params['ax_lims'])
        ax.set_ylim(params['ax_lims'])

        # Stats
        r = pearsonr(u_vals, m_vals)
        print(f'{metric} corr. coeff.: {r}')
        print(f'{metric} slope: {m}; intercept: {b}') 

    def _format_as_df(self):
        metrics = ['mean_rt', 'switch_cost', 'con_effect']
        df = pd.DataFrame(
            columns=['user', 'age_bin', 'metric', 'value', 'model_or_user'])

        for expt, ab, stats in zip(self.analysis_expt_strs,
                                   self.analysis_age_bins,
                                   self.analysis_expt_stats):
            for key in metrics:
                u_key = 'u_{0}'.format(key)
                m_key = 'm_{0}'.format(key)
                u_val = stats.summary_stats[u_key]
                m_val = stats.summary_stats[m_key]
                new_u_row = {'user': expt, 'age_bin': ab, 'metric': key, 
                             'value': u_val, 'model_or_user': 'Participants'}
                new_m_row = {'user': expt, 'age_bin': ab, 'metric': key, 
                             'value': m_val, 'model_or_user': 'Models'}
                df = df.append(new_u_row, ignore_index=True)
                df = df.append(new_m_row, ignore_index=True) 
        return df
