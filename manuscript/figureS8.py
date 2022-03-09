import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.utils import save_figure


class FigureS8():
    """Analysis methods and plotting routines to reproduce
    Figure S8 from the manuscript (comparison of behavior from
    sc+ and sc- models). 
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    behavior_keys = ['switch_cost', 'mean_rt', 'con_effect']
    figsize = (5, 2.5)
    figdpi = 300

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.A_stats = {'sc_plus_switch_cost': [], 'sc_minus_switch_cost': []}
        self.B_stats = {'sc_plus_mean_rt': [], 'sc_minus_mean_rt': []}
        self.C_stats = {'sc_plus_con_effect': [], 'sc_minus_con_effect': []}

    def make_figure(self):
        print('Making Figure S8...')
        self._run_preprocessing()
        print('Stats for Figure S8')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS8')
        print('')

    def _run_preprocessing(self):
        for expt_str, ab, uid, sc in zip(self.expts, 
                                         self.age_bins, 
                                         self.user_ids, 
                                         self.sc_status):

            if sc in ['sc+', 'sc-']:
                # Load stats from the holdout data
                stats_path = os.path.join(self.model_dir, expt_str, 
                                          self.analysis_dir, self.stats_fn)
                with open(stats_path, 'rb') as path:
                    expt_stats = pickle.load(path)

                for key, panel in zip(self.behavior_keys, [self.A_stats, 
                                                           self.B_stats,
                                                           self.C_stats]):
                    mkey = f'm_{key}'
                    if sc == 'sc+':
                        panel[f'sc_plus_{key}'].append(
                            expt_stats.summary_stats[mkey])
                    elif sc == 'sc-':
                        panel[f'sc_minus_{key}'].append(
                            expt_stats.summary_stats[mkey])

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(3, 8)

        # Switch cost
        axA = fig.add_subplot(gs[0:3, 0:2]) 
        A_params = {'key': 'switch_cost',
                    'ylabel': 'Switch cost (ms)',
                    'ylim': [-20, 190]}
        self._make_behavior_panel(axA, self.A_stats, A_params)

        # Mean RT
        axB = fig.add_subplot(gs[0:3, 3:5]) 
        B_params = {'key': 'mean_rt',
                    'ylabel': 'Mean RT (ms)',
                    'ylim': [600, 1300]}
        self._make_behavior_panel(axB, self.B_stats, B_params)

        # Congruency effect
        axC = fig.add_subplot(gs[0:3, 6:8]) 
        C_params = {'key': 'con_effect',
                    'ylabel': 'Congruency effect (ms)',
                    'ylim': [0, 230]}
        self._make_behavior_panel(axC, self.C_stats, C_params)

        return fig

    def _make_behavior_panel(self, ax, data, params):
        sc_plus = data[f"sc_plus_{params['key']}"]
        sc_minus = data[f"sc_minus_{params['key']}"]

        # Plot all models
        for plus, minus in zip(sc_plus, sc_minus):
            ax.plot([0, 1], [plus, minus], 'b-', alpha=0.1, 
                    zorder=1, linewidth=0.5)

        # Plot mean +/- s.e.m.
        m_plus = np.mean(sc_plus)
        sem_plus = np.std(sc_plus) / np.sqrt(len(sc_plus))
        m_minus = np.mean(sc_minus)
        sem_minus = np.std(sc_minus) / np.sqrt(len(sc_minus))
        self._plot_mean_sem(0, m_plus, sem_plus, ax)
        self._plot_mean_sem(1, m_minus, sem_minus, ax)

        # Adjust
        ax.set_ylabel(params['ylabel'])
        ax.set_xticks([0, 1])
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim(params['ylim'])
        ax.set_xticklabels(['sc+ models', 'sc- models'], rotation=45, 
                           ha='right', rotation_mode='anchor') 
        
        # Stats
        print(f"sc+ vs. sc- model stats for {params['key']}:")
        print(f'Mean +/- s.e.m., sc+: {m_plus} +/- {sem_plus}')
        print(f'Mean +/- s.e.m., sc-: {m_minus} +/- {sem_minus}')
        w, p = wilcoxon(sc_plus, y=sc_minus, mode='approx')
        print(f'Signed-rank test: w = {w}, p = {p}, N = {len(sc_plus)}')
        print('---------------------------------------------')

    def _plot_mean_sem(self, x, mean, sem, ax):
        ax.plot(x, mean, 'k.', markerfacecolor='k', ms=2, mew=0.5)
        ax.plot([x, x], [mean - sem, mean + sem], 'k-', linewidth=0.5)
