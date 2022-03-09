import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.utils import save_figure


class FigureS9():
    """Analysis methods and plotting routines to reproduce
    Figure S9 from the manuscript (stats on sc+ vs. sc-
    model accuracy at all noise levels and for different
    trial types). 
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'behavior_summary.pkl'
    behavior_keys = ['m_accuracy', 'con_error_rate',
                     'incon_error_rate', 'stay_error_rate', 
                     'switch_error_rate']
    noise_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
    noise_sds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    figsize = (7.5, 4)
    figdpi = 300

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
        print('Making Figure S9...')
        self._run_preprocessing()
        print('Stats for Figure S9')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS9')
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
        gs = fig.add_gridspec(5, 11)

        # Panel A: Sign-rank p-values at all noise levels
        axA = fig.add_subplot(gs[:2, 0:3])
        self._make_panel_A(axA)

        # Panel B: Error rate for congruent trials
        axB = fig.add_subplot(gs[:2, 4:7])
        B_key, B_title = 'con_error_rate', 'Congruent trials'
        self._make_error_panel(axB, B_key, B_title)

        # Panel C: Error rate for incongruent trials
        axC = fig.add_subplot(gs[:2, 8:11])
        C_key, C_title = 'incon_error_rate', 'Incongruent trials'
        self._make_error_panel(axC, C_key, C_title)

        # Panel D: Error rate for stay trials
        axD = fig.add_subplot(gs[3:, 0:3])
        D_key, D_title = 'stay_error_rate', 'Stay trials'
        self._make_error_panel(axD, D_key, D_title)

        # Panel E: Error rate for switch trials
        axE = fig.add_subplot(gs[3:, 4:7])
        E_key, E_title = 'switch_error_rate', 'Switch trials'
        self._make_error_panel(axE, E_key, E_title)

        return fig

    def _make_panel_A(self, ax):
        # Stats
        p_vals = []
        print('Panel A stats: sc+ vs. sc- accuracy, signed-rank test:')
        for n_label, n_sd in zip(self.noise_labels, self.noise_sds):
            sc_plus_vals = np.array(self.sc_plus_stats[n_label]['m_accuracy'])
            sc_minus_vals = np.array(self.sc_minus_stats[n_label]['m_accuracy'])
            _, p = wilcoxon(sc_plus_vals, y=sc_minus_vals, mode='approx')
            p_vals.append(p)
            print(f'{n_sd} SD noise: p = {p}')

        # Plot
        x = self.noise_sds
        ax.plot(x, p_vals, linestyle='-', color='b', linewidth=0.5)
        ax.plot([x[0], x[-1]], [0.05, 0.05], 'k--', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_ylabel('Signed-rank test p-value')
        ax.set_xlabel('Noise SD')

    def _make_error_panel(self, ax, key, title):
        N = 25
        ylim = [-0.03, 0.7]
        sc_plus_means = []
        sc_minus_means = []
        sc_plus_sems = []
        sc_minus_sems = []
    
        # Stats
        for n in self.noise_labels:
            sc_plus_vals = np.array(self.sc_plus_stats[n][key])
            sc_minus_vals = np.array(self.sc_minus_stats[n][key])
            sc_plus_means.append(np.mean(sc_plus_vals))
            sc_minus_means.append(np.mean(sc_minus_vals))
            sc_plus_sems.append(np.std(sc_plus_vals) / np.sqrt(N))
            sc_minus_sems.append(np.std(sc_minus_vals) / np.sqrt(N))
 
        # Plot
        ax.errorbar(self.noise_sds, sc_plus_means, yerr=sc_plus_sems, 
                    linestyle='-', color='b', label='sc+ models', linewidth=0.5)    
        ax.errorbar(self.noise_sds, sc_minus_means, yerr=sc_minus_sems, 
                    linestyle='-', color='g', label='sc- models', linewidth=0.5)    
      
        ax.set_xticks(self.noise_sds)
        ax.set_xticklabels(self.noise_sds)
        ax.set_ylabel('Error rate')
        ax.set_title(title)
        ax.legend()
        ax.get_legend().get_frame().set_linewidth(0.0)  
        ax.set_xlabel('Noise SD')
        ax.set_ylim(ylim)
