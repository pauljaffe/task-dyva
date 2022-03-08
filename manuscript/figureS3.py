import os
import pickle
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.utils import save_figure


class FigureS3():
    """Analysis methods and plotting routines to reproduce
    Figure S3 from the manuscript (model vs. participant accuracy).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'behavior_summary.pkl'
    figsize = (6, 8)

    noise_labels = ['01', '015', '02', '025', '03', '035', '04', '045',
                    '05', '055', '06']
    stats_noise = ['01', '04']
    noise_sds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    plot_x = [i for i in range(len(noise_sds))]
    plot_keys = ['accuracy', 'acc_con_effect', 'acc_switch_cost']
    plot_labels = ['Accuracy', 'Accuracy congruency effect', 
                   'Accuracy switch cost']
    
    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.stats_dict = {}
        for key in self.plot_keys:
            self.stats_dict[f'm_{key}'] = []
            self.stats_dict[f'u_{key}'] = []
        self.all_stats = {n: copy.deepcopy(self.stats_dict) 
                          for n in self.noise_labels}

    def make_figure(self):
        print('Making Figure S3...')
        self._run_preprocessing()
        print('Stats for Figure S3')
        print('-------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS3')
        print('')

    def _run_preprocessing(self):
        for expt_str, sc in zip(self.expts, 
                                self.sc_status):
            # Skip sc- models
            if sc == 'sc-':
                continue
            
            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                this_stats = pickle.load(path)
            for noise in self.noise_labels:
                for key in self.stats_dict.keys():
                    self.all_stats[noise][key].append(this_stats[noise][key])

    def _plot_figure_get_stats(self):
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        for key, label, ax in zip(self.plot_keys, self.plot_labels, axes):
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
                #ax.legend(title=None)
                ax.legend()
                ax.get_legend().get_frame().set_linewidth(0.0)   
            if key == 'acc_switch_cost':
                ax.set_xlabel('Noise SD')
            else:
                ax.set_xlabel('')

        return fig

    def _get_condition_stats(self, noise_key, noise_sd, u_key, m_key):
        u_vals = np.array(self.all_stats[noise_key][u_key])
        m_vals = np.array(self.all_stats[noise_key][m_key])
        u_mean = np.mean(u_vals)
        m_mean = np.mean(m_vals)
        u_sem = np.std(u_vals) / np.sqrt(len(u_vals))
        m_sem = np.std(m_vals) / np.sqrt(len(m_vals))
        if noise_key in self.stats_noise:
            w, p = wilcoxon(u_vals, y=m_vals)
            print(f'{u_key[2:]}, {noise_sd}SD noise stats:')
            print(f'Participant vs. model sign-rank p-val: {p}')
            print(f'Participant mean +/- s.e.m.: {u_mean} +/- {u_sem}')
            print(f'Model mean +/- s.e.m.: {m_mean} +/- {m_sem}')
            print('----------------------------------')
        return u_mean, m_mean, u_sem, m_sem
