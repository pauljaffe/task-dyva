import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.utils import save_figure, pearson_bootstrap


class Figure5():
    """Analysis methods and plotting routines to reproduce
    Figure 5 from the manuscript (separated task representations 
    confers robustness).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'summary.pkl'
    behavior_keys = ['m_accuracy', 'u_accuracy', 'u_con_effect']
    noise_labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
    noise_sds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    noise_5C = '1' # 1SD noise analyzed in panel 5C
    figsize = (7.5, 1.1)
    figdpi = 300
    palette = 'viridis'
    heatmap_palette = sns.diverging_palette(260, 10, l=50, s=100, as_cmap=True)

    def __init__(self, model_dir, save_dir, metadata, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
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
        print('Making Figure 5...')
        self._run_preprocessing()
        print('Stats for Figure 5')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'Fig5')
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
                for key in self.behavior_keys:
                    stat = this_stats[n][key]
                    if sc == 'sc+':
                        self.sc_plus_stats[n][key].append(stat)
                    elif sc == 'sc-':
                        self.sc_minus_stats[n][key].append(stat)

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(4, 9, wspace=2)

        # Panel A: Accuracy vs. noise summary
        axA = fig.add_subplot(gs[:, :3])
        self._make_panel_A(axA)

        # Panel B: Delta accuracy heatmap
        axB = fig.add_subplot(gs[:, 3:6])
        self._make_panel_B(axB)

        # Panel C: Model robustness to noise vs. congruency effect
        axC = fig.add_subplot(gs[:, 6:8])
        self._make_panel_C(axC)

        return fig

    def _make_panel_A(self, ax):
        cmap = sns.color_palette(self.palette, as_cmap=True)
        N = 25 # number of models
        yticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ylims = [0.38, 1.05]
        xticks = [0]
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
        print('Panel A stats:')
        for n_label, n_sd in zip(self.noise_labels, self.noise_sds):
            sc_plus_vals = np.array(self.sc_plus_stats[n_label]['m_accuracy'])
            sc_minus_vals = np.array(self.sc_minus_stats[n_label]['m_accuracy'])
            sc_plus_means = np.append(sc_plus_means, np.mean(sc_plus_vals))
            sc_minus_means = np.append(sc_minus_means, np.mean(sc_minus_vals))
            sc_plus_errors = np.append(sc_plus_errors, 
                                       np.std(sc_plus_vals) / np.sqrt(N))
            sc_minus_errors = np.append(sc_minus_errors, 
                                        np.std(sc_minus_vals) / np.sqrt(N))
            wstat, p = wilcoxon(sc_plus_vals, sc_minus_vals, mode='approx')
            print(f'Sign-rank test at {n_sd}SD noise: w = {wstat}, p = {p}')
     
        # Plot
        sns.boxplot(y=u_vals, ax=ax, orient='v', fliersize=1,
                    color='mediumorchid', linewidth=0.5, width=0.05)
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
        ax.plot([0.065, 0.065], [0.4, 1.03], 'k--', linewidth=0.25)

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
        ax.set_xlim([-0.075, 1.05])
        print('------------------------------')

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
        plot_noise = '1'
        plot_noise_label = '1'
        line_ext = 0

        print('Stats for panel C:')
        xlab = 'Participant congruency effect (ms)'
        ylab = f'Model robustness to noise'
        x = self.acc_diff_stats[plot_noise]['u_con_effect']
        y = self.acc_diff_stats[plot_noise]['acc_diff']
        _ = plot_scatter2(x, y, ax, line_ext, xlab, ylab, self.rng,
                          n_boot=self.n_boot, alpha=self.alpha,
                          text_x=0.65, text_y=0.15)
        ax.set_xticks([0, 50, 100, 150, 200, 250])


def plot_scatter2(d1, d2, ax, line_ext, xlabel, ylabel, rng,
                  n_boot=1000, alpha=0.05,
                  plot_unity=False, text_x=0.05, text_y=0.95):
    # Plot best fit line
    plot_x = np.array([min(d1) - line_ext, 
                       max(d1) + line_ext])
    m, b = np.polyfit(d1, d2, 1)
    ax.plot(plot_x, m * plot_x + b, 'r--', zorder=1, linewidth=0.5)

    # Plot unity line
    if plot_unity:
        ax.plot(plot_x, plot_x, 'k-', zorder=1, linewidth=0.5)
    
    # Plot all individuals
    ax.scatter(d1, d2, s=0.5, marker='o', zorder=2, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Stats
    r, p, ci_lo, ci_hi = pearson_bootstrap(d1, d2, rng,
                                           n_boot=n_boot,
                                           alpha=alpha)
    p_str = '{:0.2e}'.format(p)
    tstr = f'r = {round(r, 2)}, 95% CI: ({round(ci_lo, 2)}, {round(ci_hi, 2)})\np = {p_str}'
    print(tstr)

    return r
