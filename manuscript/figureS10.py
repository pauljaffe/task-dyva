import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.utils import save_figure, pearson_bootstrap, pj_bootstrap


class FigureS10():
    """Analysis methods and plotting routines to reproduce
    Figure S10 from the manuscript (stats on sc+ vs. sc-
    model accuracy at all noise levels and for different
    trial types). 
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'summary.pkl'
    behavior_keys = ['m_accuracy', 'con_error_rate',
                     'incon_error_rate', 'stay_error_rate', 
                     'switch_error_rate']
    noise_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
    noise_sds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    figsize = (7.5, 4)
    figdpi = 300

    def __init__(self, model_dir, save_dir, metadata, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.sc_status = metadata['switch_cost_type']
        self.exgauss = metadata['exgauss']
        self.early = metadata['early']
        self.optimal = metadata['optimal']
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05

        # Containers for summary stats
        stat_dict = {k: [] for k in self.behavior_keys}
        self.sc_minus_stats = {n: copy.deepcopy(stat_dict) 
                               for n in self.noise_labels}
        self.sc_plus_stats = copy.deepcopy(self.sc_minus_stats)
        diff_dict = {'acc_diff': [], 'u_con_effect': []}
        self.acc_diff_stats = {n: copy.deepcopy(diff_dict)
                               for n in self.noise_labels}

    def make_figure(self):
        print('Making Figure S10...')
        self._run_preprocessing()
        print('Stats for Figure S10')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS10')
        print('')

    def _run_preprocessing(self):
        for expt_str, ab, uid, sc, exg, early, opt in zip(self.expts, 
                                                          self.age_bins, 
                                                          self.user_ids, 
                                                          self.sc_status,
                                                          self.exgauss,
                                                          self.early,
                                                          self.optimal):

            # Skip exgauss+ models, early models, optimal models
            if exg == 'exgauss+' or early or opt:
                continue

            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                this_stats = pickle.load(path)

            for n in self.noise_labels:
                if n == '01':
                    acc_01 = this_stats[n]['m_accuracy']
                acc_diff = acc_01 - this_stats[n]['m_accuracy']
                if sc != 'sc-':
                    self.acc_diff_stats[n]['acc_diff'].append(acc_diff)
                    self.acc_diff_stats[n]['u_con_effect'].append(
                        this_stats[n]['u_con_effect'])

                if sc in ['sc+', 'sc-']:
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

        # Panel A: Error rate for congruent trials
        axA = fig.add_subplot(gs[:2, :3])
        A_key, A_title = 'con_error_rate', 'Congruent trials'
        self._make_error_panel(axA, A_key, A_title)

        # Panel B: Error rate for incongruent trials
        axB = fig.add_subplot(gs[:2, 4:7])
        B_key, B_title = 'incon_error_rate', 'Incongruent trials'
        self._make_error_panel(axB, B_key, B_title)

        # Panel C: Error rate for stay trials
        axC = fig.add_subplot(gs[:2, 8:11])
        C_key, C_title = 'stay_error_rate', 'Stay trials'
        self._make_error_panel(axC, C_key, C_title)

        # Panel D: Error rate for switch trials
        axD = fig.add_subplot(gs[3:, :3])
        D_key, D_title = 'switch_error_rate', 'Switch trials'
        self._make_error_panel(axD, D_key, D_title)

        # Panel E: Robustness vs. congruency effect
        axE = fig.add_subplot(gs[3:, 4:7])
        self._make_panel_E(axE)

        return fig

    def _make_error_panel(self, ax, key, title):
        N = 25
        ylim = [-0.03, 0.7]
        sc_plus_means = []
        sc_minus_means = []
        sc_plus_cis = np.zeros((2, len(self.noise_labels)))
        sc_minus_cis = np.zeros((2, len(self.noise_labels)))
    
        # Stats
        for i, n in enumerate(self.noise_labels):
            sc_plus_vals = np.array(self.sc_plus_stats[n][key])
            sc_minus_vals = np.array(self.sc_minus_stats[n][key])
            m_sc_plus, lo_sc_plus, hi_sc_plus = pj_bootstrap(sc_plus_vals,
                                                             self.rng, 
                                                             self.n_boot,
                                                             self.alpha,
                                                             ci_for_plot=True)
            m_sc_minus, lo_sc_minus, hi_sc_minus = pj_bootstrap(sc_minus_vals,
                                                                self.rng, 
                                                                self.n_boot,
                                                                self.alpha,
                                                                ci_for_plot=True)
            sc_plus_means.append(m_sc_plus)
            sc_minus_means.append(m_sc_minus)
            sc_plus_cis[0, i] = lo_sc_plus
            sc_plus_cis[1, i] = hi_sc_plus
            sc_minus_cis[0, i] = lo_sc_minus
            sc_minus_cis[1, i] = hi_sc_minus
 
        # Plot
        ax.errorbar(self.noise_sds, sc_plus_means, yerr=sc_plus_cis,
                    linestyle='-', color='b', label='sc+ models', linewidth=0.5)    
        ax.errorbar(self.noise_sds, sc_minus_means, yerr=sc_minus_cis, 
                    linestyle='-', color='g', label='sc- models', linewidth=0.5)    
      
        ax.set_xticks(self.noise_sds)
        ax.set_xticklabels(self.noise_sds)
        ax.set_ylabel('Error rate')
        ax.set_title(title)
        ax.legend()
        ax.get_legend().get_frame().set_linewidth(0.0)  
        ax.set_xlabel('Noise SD')
        ax.set_ylim(ylim)

    def _make_panel_E(self, ax):
        rvals = []
        N = len(self.noise_labels[1:])
        cis = np.zeros((2, N))
        print('Stats for diff. in model accuracy vs. participant congruency effect')
        print('----------------------------------------')
        for ind, pn, pnlab in zip(range(N), self.noise_sds[1:], 
                                  self.noise_labels[1:]):
            x = self.acc_diff_stats[pnlab]['u_con_effect']
            y = self.acc_diff_stats[pnlab]['acc_diff']
            r, p, ci_lo, ci_hi = pearson_bootstrap(x, y, self.rng,
                                                   n_boot=self.n_boot,
                                                   alpha=self.alpha)
            rvals.append(r)
            cis[0, ind] = r - ci_lo
            cis[1, ind] = ci_hi - r
            print(f'Stats for noise = {pn}SD:')
            p_str = '{:0.2e}'.format(p)
            tstr = f'r = {round(r, 2)}, 95% CI: ({round(ci_lo, 2)}, {round(ci_hi, 2)})\np = {p_str}'
            print(tstr)

        ax.errorbar(self.noise_sds[1:], rvals, yerr=cis, linestyle='-',
                    linewidth=0.5)
        ax.set_xlabel('Noise SD')
        ax.set_xticks(self.noise_sds[1:])
        ax.set_xticklabels(self.noise_sds[1:])
        ax.set_ylabel("Pearson's r")
        print('-------------------------------')
