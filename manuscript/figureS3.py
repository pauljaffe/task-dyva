import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.visualization import BarPlot
from task_dyva.utils import save_figure, plot_scatter, expt_stats_to_df


class FigureS3():
    """Analysis methods and plotting routines to reproduce
    Figure S3 from the manuscript (model vs. participant accuracy;
    model vs. participant RT variability).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'summary.pkl'
    outputs_fn = 'holdout_outputs_01SD.pkl'
    figsize = (7, 3.5)
    figdpi = 300

    noise_labels = ['01', '015', '02', '025', '03', '035', '04', '045',
                    '05', '055', '06']
    stats_noise = ['01', '04']
    noise_sds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    plot_x = [i for i in range(len(noise_sds))]
    plot_keys = ['accuracy', 'acc_con_effect', 'acc_switch_cost']
    plot_labels = ['Accuracy', 'Accuracy congruency effect', 
                   'Accuracy switch cost']
    age_bin_labels = ['20-29', '30-39', '40-49', '50-59', 
                      '60-69', '70-79', '80-89']
    line_ext = 0.1
    
    def __init__(self, model_dir, save_dir, metadata, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.sc_status = metadata['switch_cost_type']
        self.exgauss = metadata['exgauss']
        self.early = metadata['early']
        self.optimal = metadata['optimal']
        self.age_bins = metadata['age_range']
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05

        # Containers for summary stats
        self.stats_dict = {}
        for key in self.plot_keys:
            self.stats_dict[f'm_{key}'] = []
            self.stats_dict[f'u_{key}'] = []
        self.acc_stats = {n: copy.deepcopy(self.stats_dict) 
                          for n in self.noise_labels}
        self.var_stats = {'u_rt_sd': [], 'm_rt_sd': []}
        self.analysis_expt_stats = []
        self.analysis_age_bins = []
        self.analysis_expt_strs = []

    def make_figure(self):
        print('Making Figure S3...')
        self._run_preprocessing()
        print('Stats for Figure S3')
        print('-------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS3')
        print('')

    def _run_preprocessing(self):
        for expt_str, ab, uid, sc, exg, early, opt in zip(self.expts, 
                                                          self.age_bins, 
                                                          self.user_ids, 
                                                          self.sc_status,
                                                          self.exgauss,
                                                          self.early,
                                                          self.optimal):

            # Skip sc- models, exgauss+ models, early models, optimal models
            if sc == 'sc-' or exg == 'exgauss+' or early or opt:
                continue
            
            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                this_stats = pickle.load(path)
            for noise in self.noise_labels:
                for key in self.stats_dict.keys():
                    self.acc_stats[noise][key].append(this_stats[noise][key])

            # Load data for variability analyses
            var_path = os.path.join(self.model_dir, expt_str, 
                                    self.analysis_dir, self.outputs_fn)
            with open(var_path, 'rb') as path:
                var_expt_stats = pickle.load(path)
            self.analysis_age_bins.append(ab)
            self.analysis_expt_stats.append(var_expt_stats)
            self.analysis_expt_strs.append(expt_str)
            for key in self.var_stats.keys():
                self.var_stats[key].append(var_expt_stats.summary_stats[key])

    def _plot_figure_get_stats(self):
        fig, axes = plt.subplots(2, 3, figsize=self.figsize,
                                 dpi=self.figdpi)

        # Accuracy panels: a-c
        for key, label, ax_ind in zip(self.plot_keys, self.plot_labels, range(3)):
            ax = axes[0, i]
            u_key = f'u_{key}'
            m_key = f'm_{key}'
            this_u_means = []
            this_m_means = []
            this_u_sems = []
            this_m_sems = []
            
            for noise_key, noise_sd in zip(self.noise_labels, self.noise_sds):
                u_mean, m_mean, u_sem, m_sem = self._get_condition_stats(
                    noise_key, noise_sd, u_key, m_key)
                this_u_means.append(u_mean)
                this_m_means.append(m_mean)
                this_u_sems.append(u_sem)
                this_m_sems.append(m_sem)
            
            ax.errorbar(self.plot_x, this_u_means, yerr=this_u_sems, 
                        linestyle='-', color='b', label='Participants', 
                        linewidth=0.5)
            ax.errorbar(self.plot_x, this_m_means, yerr=this_m_sems, 
                        linestyle='-', color='g', label='Models', 
                        linewidth=0.5)    
            ax.set_xticks(self.plot_x)
            ax.set_xticklabels(self.noise_sds)
            ax.set_ylabel(label)
            if key == 'accuracy':
                ax.legend()
                ax.get_legend().get_frame().set_linewidth(0.0)   
            if key == 'acc_con_effect':
                ax.set_xlabel('Noise SD')
            else:
                ax.set_xlabel('')

        # Panel d: Model vs. participant scatter for RT SD
        D_params = {'ax_lims': [20, 350],
                    'metric': 'rt_sd',
                    'label': 'RT SD (ms)'}
        plot_scatter(self.var_stats, D_params, axes[1, 0], self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha)

        # Panel e: Model vs. participant RT SD binned by age
        error_type = 'sem'
        stats_df = expt_stats_to_df(['rt_sd'],
                                    self.analysis_expt_strs,
                                    self.analysis_age_bins,
                                    self.analysis_expt_stats)
        E_params = {'ylabel': 'RT SD (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': True}
        E_bar = BarPlot(stats_df)
        E_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                               error_type, axes[1, 1], **E_params)
        axes[1, 1].set_xlabel('Age bin (years)')

        plt.tight_layout()

        return fig

    def _get_condition_stats(self, noise_key, noise_sd, u_key, m_key):
        u_vals = np.array(self.acc_stats[noise_key][u_key])
        m_vals = np.array(self.acc_stats[noise_key][m_key])
        u_mean = np.mean(u_vals)
        m_mean = np.mean(m_vals)
        u_sem = np.std(u_vals) / np.sqrt(len(u_vals))
        m_sem = np.std(m_vals) / np.sqrt(len(m_vals))
        if noise_key in self.stats_noise:
            w, p = wilcoxon(u_vals, y=m_vals, mode='approx')
            print(f'{u_key[2:]}, {noise_sd}SD noise stats:')
            print(f'Participant vs. model signed-rank p-val: {p}')
            print(f'Participant mean +/- s.e.m.: {u_mean} +/- {u_sem}')
            print(f'Model mean +/- s.e.m.: {m_mean} +/- {m_sem}')
            print('----------------------------------')
        return u_mean, m_mean, u_sem, m_sem
