import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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
    figsize = (6.5, 1)
    figdpi = 300
    palette = 'viridis'

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
        gs = fig.add_gridspec(4, 8, wspace=20, hspace=2)

        # Panel A: Accuracy vs. noise summary
        axA = fig.add_subplot(gs[:, 0:3])
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

        u_means = np.array([])
        sc_plus_means = np.array([])
        sc_minus_means = np.array([])
        u_errors = np.array([])
        sc_plus_errors = np.array([])
        sc_minus_errors = np.array([])
        
        # Stats
        print('Accuracy vs. noise stats, sc+ vs. sc- models, signed-rank test:')
        for n_label, n_sd in zip(self.noise_labels, self.noise_sds):
            # Note: participant accuracy does not vary across noise levels
            u_vals = np.array(self.sc_plus_stats[n_label]['u_accuracy'])
            sc_plus_vals = np.array(self.sc_plus_stats[n_label]['m_accuracy'])
            sc_minus_vals = np.array(self.sc_minus_stats[n_label]['m_accuracy'])

            u_means = np.append(u_means, np.mean(u_vals))
            sc_plus_means = np.append(sc_plus_means, np.mean(sc_plus_vals))
            sc_minus_means = np.append(sc_minus_means, np.mean(sc_minus_vals))

            u_errors = np.append(u_errors, np.std(u_vals) / np.sqrt(N))
            sc_plus_errors = np.append(sc_plus_errors, 
                                       np.std(sc_plus_vals) / np.sqrt(N))
            sc_minus_errors = np.append(sc_minus_errors, 
                                        np.std(sc_minus_vals) / np.sqrt(N))

            # Print stats
            _, p = wilcoxon(sc_plus_vals, y=sc_minus_vals)
            print(f'{n_sd} SD noise: p = {p}')
            p_key = f'{nc}_p_val'
     
        # Plot
        ax.plot(self.noise_sds, u_means, linestyle='-', color='k', 
                label='Participants', linewidth=0.5)
        ax.plot(self.noise_sds, sc_plus_means, linestyle='-', 
                color=cmap(0.3), label='sc+ models', linewidth=0.5)    
        ax.plot(self.noise_sds, sc_minus_means, linestyle='-', 
                color=cmap(0.7), label='sc- models', linewidth=0.5)    
        ax.fill_between(self.noise_sds, u_means - u_errors, u_means + u_errors, 
                        alpha=0.2, facecolor='k', label=None)
        ax.fill_between(self.noise_sds, sc_plus_means - sc_plus_errors, 
                        sc_plus_means + sc_plus_errors, alpha=0.2, 
                        facecolor=cmap(0.3), label=None)
        ax.fill_between(self.noise_sds, sc_minus_means - sc_minus_errors, 
                        sc_minus_means + sc_minus_errors, alpha=0.2, 
                        facecolor=cmap(0.3), label=None)
          
        ax.set_xticks(self.noise_sds)
        ax.set_xticklabels(self.noise_sds)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylabel('Accuracy')
        ax.legend(title=None)
        ax.get_legend().get_frame().set_linewidth(0.0)  
        ax.legend(framealpha=0)
        ax.set_xlabel('Noise SD')

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
        sns.heatmap(delta_np, center=0, cmap=palette, ax=ax)
        ax.set_xticks(self.noise_sds)
        ax.set_xticklabels(self.noise_sds)
        ax.set_yticks([])
        ax.tick_params(axis='both')
        ax.set_ylabel('Models')
        ax.legend(title=None)
        ax.get_legend().get_frame().set_linewidth(0.0)  
        ax.legend(framealpha=0)
        ax.set_xlabel('Noise SD')

    def _make_panel_C(self, ax):
        keys = ['con_error_rate', 'incon_error_rate', 'stay_error_rate', 
                'switch_error_rate']
        df_keys = ['Congruent', 'Incongruent', 'Stay', 'Switch']

        # Reformat as data frame
        all_dfs = []
        for key, df_key in zip(keys, df_keys):
            sc_plus = pd.DataFrame({'trial_type': df_key, 'model_type': 'sc+',
                                    'p_error': self.sc_plus_stats[self.noise_5C][key]})
            sc_minus = pd.DataFrame({'trial_type': df_key, 'model_type': 'sc-',
                                     'p_error': self.sc_minus_stats[self.noise_5C][key]})
            all_dfs.append(sc_plus)
            all_dfs.append(sc_minus)
        df = pd.concat(all_dfs)

        # Plot
        params = {'ylim': [0, 0.45],
                  'ylabel': 'Conditional error rate',
                  'xticklabels': df_keys,
                  'plot_legend': True}
        error_type = 'sem'
        bar = BarPlot(df)
        _ = bar.plot_grouped_bar('trial_type', 'p_error', 'model_type',
                                 error_type, ax, **params)
