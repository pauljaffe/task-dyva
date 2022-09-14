import os
import pickle
import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, ranksums

from task_dyva.utils import save_figure, plot_scatter


class FigureS5():
    """Analysis for Figure S5: models trained on early practice data
    and with exGauss response kernels.
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    line_ext = 10
    figsize = (7, 6.5)
    figdpi = 300
    palette = 'viridis'

    def __init__(self, model_dir, save_dir, metadata, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.metadata = metadata
        self.expts = metadata['name']
        self.user_ids = metadata['user_id']
        self.exgauss = metadata['exgauss']
        self.early = metadata['early']
        self.sc_status = metadata['switch_cost_type']
        self.optimal = metadata['optimal']
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05
        self.hist_bw = 20  # ms
        self.eps = 1e-6  # small constant to add to PDFs for KLD calc

        # Containers for summary stats
        self.early_stats = {'m_switch_cost': [], 'u_switch_cost': [],
                            'm_mean_rt': [], 'u_mean_rt': [], 
                            'm_con_effect': [], 'u_con_effect': []}
        self.exgaussian_stats = {'m_switch_cost': [], 'u_switch_cost': [],
                            'm_mean_rt': [], 'u_mean_rt': [], 
                            'm_con_effect': [], 'u_con_effect': []}
        self.gaussian_stats = {'m_switch_cost': [], 'u_switch_cost': [],
                            'm_mean_rt': [], 'u_mean_rt': [], 
                            'm_con_effect': [], 'u_con_effect': []}
        self.exgauss_klds = []
        self.gauss_klds = []

    def make_figure(self):
        print('Making Figure S5...')
        self._run_preprocessing()
        print('Stats for Figure S5')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS5')
        print('')

    def _run_preprocessing(self):
        for expt_str, uid, exg, earl, sc, opt in zip(self.expts, 
                                                     self.user_ids, 
                                                     self.exgauss,
                                                     self.early,
                                                     self.sc_status,
                                                     self.optimal):

            if sc == 'sc-' or opt:
                continue

            if earl:  # Early practice models
                _ = self._add_user_stats(expt_str, 'early')
                continue
            elif exg == 'exgauss+':  # Models trained with exGauss kernel
                stats = self._add_user_stats(expt_str, 'exgauss+')
                user_rts = stats.df['urt_ms'].values
            else:  # Models trained with Gaussian kernel (excluding sc- and early models)
                stats = self._add_user_stats(expt_str, 'exgauss-')
                user_rts = stats.df['urt_ms'].values
            model_rts = stats.df['mrt_ms'].values

            # Get KLDs
            bin_edges = self._kld_bins(user_rts)
            model_hist = np.histogram(model_rts, bins=bin_edges, density=True)[0]
            user_hist = np.histogram(user_rts, bins=bin_edges, density=True)[0]
            kld = entropy(model_hist, user_hist+self.eps)
            if exg == 'exgauss+':
                self.exgauss_klds.append(kld)
            else:
                self.gauss_klds.append(kld)

    def _kld_bins(self, user_rts):
        min_rt = user_rts.min()
        max_rt = user_rts.max()
        bin_min = self.hist_bw * (min_rt // self.hist_bw) - self.hist_bw/2
        bin_max = self.hist_bw * (max_rt // self.hist_bw) + 1.5*self.hist_bw
        bin_edges = np.arange(bin_min, bin_max, self.hist_bw)
        return bin_edges

    def _add_user_stats(self, expt_str, model_type):
        stats_path = os.path.join(self.model_dir, expt_str, 
                                  self.analysis_dir, self.stats_fn)
        with open(stats_path, 'rb') as path:
            expt_stats = pickle.load(path)
        for key in self.exgaussian_stats.keys():
            val = expt_stats.summary_stats[key]
            if model_type == 'exgauss+':
                self.exgaussian_stats[key].append(val)
            elif model_type == 'exgauss-':
                self.gaussian_stats[key].append(val)
            elif model_type == 'early':
                self.early_stats[key].append(val)
        return expt_stats

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(26, 25)

        print('Models trained on early practice data:')
        N_early = len(self.early_stats['u_mean_rt'])
        print(f'N early practice models: {N_early}')
        # Panel a: Mean RT, early pracice
        A_params = {'ax_lims': [600, 1700],
                    'metric': 'mean_rt',
                    'label': 'mean RT (ms)'}
        A_ax = fig.add_subplot(gs[:7, :7])
        plot_scatter(self.early_stats, A_params, A_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha,
                     plot_stats=True)

        # Panel b: Switch cost, early practice
        B_params = {'ax_lims': [-50, 500],
                    'metric': 'switch_cost',
                    'label': 'switch cost (ms)'}
        B_ax = fig.add_subplot(gs[:7, 9:16])
        plot_scatter(self.early_stats, B_params, B_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha,
                     plot_stats=True)
        B_ax.set_title('Early practice data models', fontsize=7)
        
        # Panel c: Congruency effect, early practice
        C_params = {'ax_lims': [0, 280],
                    'metric': 'con_effect',
                    'label': 'congruency effect (ms)'}
        C_ax = fig.add_subplot(gs[:7, 18:25])
        plot_scatter(self.early_stats, C_params, C_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha,
                     plot_stats=True)

        print('ExGaussian smoothing stats:')
        N_exgauss = len(self.exgaussian_stats['u_mean_rt'])
        N_gauss = len(self.gaussian_stats['u_mean_rt'])
        print(f'N exGaussian models: {N_exgauss}')
        print(f'N Gaussian models: {N_gauss}')
        # Panel d: Mean RT, exGaussian smoothing
        D_params = {'ax_lims': [600, 1300],
                    'metric': 'mean_rt',
                    'label': 'mean RT (ms)'}
        D_ax = fig.add_subplot(gs[10:17, :7])
        plot_scatter(self.exgaussian_stats, D_params, D_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha,
                     plot_stats=True)

        # Panel e: Switch cost, exGaussian smoothing
        E_params = {'ax_lims': [-25, 200],
                    'metric': 'switch_cost',
                    'label': 'switch cost (ms)'}
        E_ax = fig.add_subplot(gs[10:17, 9:16])
        plot_scatter(self.exgaussian_stats, E_params, E_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha,
                     plot_stats=True)
        E_ax.set_title('exGaussian smoothing kernel models', fontsize=7)
        
        # Panel f: Congruency effect, exGaussian smoothing
        F_params = {'ax_lims': [0, 200],
                    'metric': 'con_effect',
                    'label': 'congruency effect (ms)'}
        F_ax = fig.add_subplot(gs[10:17, 18:25])
        plot_scatter(self.exgaussian_stats, F_params, F_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha,
                     plot_stats=True)

        # Panel g: Participant vs. model RT distribution KL-divergence (CDF)
        G_ax = fig.add_subplot(gs[19:, :7])
        sns.ecdfplot(data=self.exgauss_klds, ax=G_ax, color='b', 
                     label='exGaussian', linewidth=0.5)
        sns.ecdfplot(data=self.gauss_klds, ax=G_ax, color='g', 
                     label='Gaussian', linewidth=0.5)
        rs_stat, rs_p = ranksums(self.exgauss_klds, self.gauss_klds)
        exgauss_mean = np.mean(self.exgauss_klds)
        gauss_mean = np.mean(self.gauss_klds)
        exgauss_sem = np.std(self.exgauss_klds) / np.sqrt(len(self.exgauss_klds))
        gauss_sem = np.std(self.gauss_klds) / np.sqrt(len(self.gauss_klds))
        print(f'Exgauss KLD mean +/- s.e.m.: {exgauss_mean} +/- {exgauss_sem}')
        print(f'Gaussian KLD mean +/- s.e.m.: {gauss_mean} +/- {gauss_sem}')
        print(f'KLD ranksum test stat: {rs_stat}, p-val: {rs_p}')
        G_ax.set_xlim(0, 0.6)
        G_ax.set_xlabel('Participant vs. model $D_{KL}$')
        G_ax.set_ylabel('Proportion')
        G_ax.legend()
        G_ax.get_legend().get_frame().set_linewidth(0.0)

        return fig
