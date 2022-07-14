import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.visualization import BarPlot
from task_dyva.utils import save_figure


class Figure5():
    """Analysis methods and plotting routines to reproduce
    Figure 5 from the manuscript (separated task representations 
    confers robustness).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'behavior_summary.pkl'
    behavior_keys = ['m_accuracy', 'u_accuracy', 'con_error_rate',
                     'incon_error_rate', 'stay_error_rate', 
                     'switch_error_rate']
    noise_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
    noise_sds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    noise_5C = '05' # 0.5SD noise analyzed in panel 5C
    figsize = (7.5, 1.5)
    figdpi = 300
    palette = 'viridis'
    heatmap_palette = sns.diverging_palette(260, 10, l=50, s=100, as_cmap=True)

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        stat_dict = {k: [] for k in self.behavior_keys}
        self.sc_minus_stats = {n: copy.deepcopy(stat_dict) 
                               for n in self.noise_labels}
        self.sc_plus_stats = copy.deepcopy(self.sc_minus_stats)

    def make_figure(self):
        print('Making Figure 5...')
        self._run_preprocessing()
        print('Stats for Figure 5')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'Fig5')
        print('')

    def _run_preprocessing(self):
        for expt_str, sc in zip(self.expts, 
                                self.sc_status):

            if sc in ['sc+', 'sc-']:
                stats_path = os.path.join(self.model_dir, expt_str, 
                                          self.analysis_dir, self.stats_fn)
                with open(stats_path, 'rb') as path:
                    this_stats = pickle.load(path)

                for n in self.noise_labels:
                    for key in self.behavior_keys:
                        stat = this_stats[n][key]
                        if sc == 'sc+':
                            self.sc_plus_stats[n][key].append(stat)
                        elif sc == 'sc-':
                            self.sc_minus_stats[n][key].append(stat)

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(4, 8, wspace=2)

        # Panel A: Accuracy vs. noise summary
        axA = fig.add_subplot(gs[:, :3])
        self._make_panel_A(axA)

        # Panel B: Delta accuracy heatmap
        axB = fig.add_subplot(gs[:, 3:6])
        self._make_panel_B(axB)

        # Panel C: Error rate by trial type
        axC = fig.add_subplot(gs[:, 6:])
        self._make_panel_C(axC)

        return fig

    def _make_panel_A(self, ax):
        cmap = sns.color_palette(self.palette, as_cmap=True)
        N = 25 # number of models
        yticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ylims = [0.38, 1.05]
        xticks = [-0.1]
        xticks.extend(self.noise_sds)
        xticklabels = ['Participants']
        xticklabels.extend(self.noise_sds)

        u_means = np.array([])
        sc_plus_means = np.array([])
        sc_minus_means = np.array([])
        u_errors = np.array([])
        sc_plus_errors = np.array([])
        sc_minus_errors = np.array([])
        
        # Stats
        u_vals = np.array(self.sc_plus_stats['01']['u_accuracy'])
        u_mean = np.mean(u_vals)
        u_error = np.std(u_vals) / np.sqrt(N)
        for n_label, n_sd in zip(self.noise_labels, self.noise_sds):
            sc_plus_vals = np.array(self.sc_plus_stats[n_label]['m_accuracy'])
            sc_minus_vals = np.array(self.sc_minus_stats[n_label]['m_accuracy'])
            sc_plus_means = np.append(sc_plus_means, np.mean(sc_plus_vals))
            sc_minus_means = np.append(sc_minus_means, np.mean(sc_minus_vals))
            sc_plus_errors = np.append(sc_plus_errors, 
                                       np.std(sc_plus_vals) / np.sqrt(N))
            sc_minus_errors = np.append(sc_minus_errors, 
                                        np.std(sc_minus_vals) / np.sqrt(N))
     
        # Plot
        ax.bar(-0.1, u_mean, yerr=u_error, width=0.05, color='plum',
               error_kw={'elinewidth': 1})
        ax.plot(self.noise_sds, sc_plus_means, linestyle='-', 
                 color=cmap(0.3), label='sc+ models', linewidth=0.5)    
        ax.plot(self.noise_sds, sc_minus_means, linestyle='-', 
                 color=cmap(0.7), label='sc- models', linewidth=0.5)    
        ax.fill_between(self.noise_sds, sc_plus_means - sc_plus_errors, 
                         sc_plus_means + sc_plus_errors, alpha=0.2, 
                         facecolor=cmap(0.3), label=None)
        ax.fill_between(self.noise_sds, sc_minus_means - sc_minus_errors, 
                         sc_minus_means + sc_minus_errors, alpha=0.2, 
                         facecolor=cmap(0.7), label=None)
        ax.plot([0, 0], [0.4, 1.03], 'k--', linewidth=0.5)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        tick1 = ax.get_xticklabels()[0]
        tick1.set_rotation(45)
        ax.set_yticks(yticks)
        ax.legend(title=None)
        ax.get_legend().get_frame().set_linewidth(0.0)  
        ax.legend(framealpha=0)
        ax.set_xlabel('Noise SD')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(ylims)

    def _make_panel_B(self, ax):
        # Calculate delta accuracy for each model
        N = 25 # number of models
        deltas = []
        for i in range(N):
            model_deltas = []
            for n in self.noise_labels:
                sc_plus_acc = self.sc_plus_stats[n]['m_accuracy'][i]
                sc_minus_acc = self.sc_minus_stats[n]['m_accuracy'][i]
                model_deltas.append(sc_minus_acc - sc_plus_acc)
            deltas.append(model_deltas)
        delta_np = np.stack(deltas)

        # Plot
        plot_x = [i + 0.5 for i in range(len(self.noise_labels))]
        sns.heatmap(delta_np, center=0, cmap=self.heatmap_palette, ax=ax,
                    vmin=-0.4, vmax=0.4, cbar_kws={'ticks': [-0.25, 0, 0.25]})
                    
        ax.set_xticks(plot_x)
        ax.set_xticklabels(self.noise_sds)
        ax.set_yticks([])
        ax.set_ylabel('Models')
        ax.set_xlabel('Noise SD')

    def _make_panel_C(self, ax):
        print('Panel C stats: sc+ vs. sc- error rates, signed-rank test:')
        keys = ['con_error_rate', 'incon_error_rate', 'stay_error_rate', 
                'switch_error_rate']
        df_keys = ['Congruent', 'Incongruent', 'Stay', 'Switch']

        # Reformat as data frame, print stats
        all_dfs = []
        for key, df_key in zip(keys, df_keys):
            plus_data = self.sc_plus_stats[self.noise_5C][key]
            minus_data = self.sc_minus_stats[self.noise_5C][key]
            plus_df = pd.DataFrame({'trial_type': df_key, 'model_type': 'sc+',
                                    'p_error': plus_data})
            minus_df = pd.DataFrame({'trial_type': df_key, 'model_type': 'sc-',
                                     'p_error': minus_data})
            all_dfs.append(plus_df)
            all_dfs.append(minus_df)

            _, p = wilcoxon(plus_data, y=minus_data, mode='approx')
            print(f'{df_key} error rate: p = {p}')
            print(f'sc+ mean: {np.mean(plus_data)}; sc- mean: {np.mean(minus_data)}')
        print('----------------------------')
        df = pd.concat(all_dfs)

        # Plot
        params = {'ylim': [0, 0.45],
                  'ylabel': 'Error rate',
                  'xticklabels': df_keys,
                  'plot_legend': True,
                  'elinewidth': 0.5}
        error_type = 'sem'
        bar = BarPlot(df)
        _ = bar.plot_grouped_bar('trial_type', 'p_error', 'model_type',
                                 error_type, ax, **params)
