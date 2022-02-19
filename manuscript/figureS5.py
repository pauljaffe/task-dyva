import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from task_dyva.utils import save_figure, plot_scatter


class FigureS5():
    """Analysis methods and plotting routines to reproduce
    Figure S5 from the manuscript (behavior summary at higher stimulus noise).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_04SD.pkl'
    figsize = (16, 4.5)
    line_ext = 10

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.group_stats = {'m_switch_cost': [], 'u_switch_cost': [],
                            'm_mean_rt': [], 'u_mean_rt': [], 
                            'm_con_effect': [], 'u_con_effect': [], 
                            'u_rt_sd': [], 'm_rt_sd': []}

    def make_figure(self):
        print('Making Figure S5...')
        self._run_preprocessing()
        print('Stats for Figure S5')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS5')
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
            for key in self.group_stats.keys():
                self.group_stats[key].append(expt_stats.summary_stats[key])

    def _plot_figure_get_stats(self):
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Panel A: Mean RT scatter
        A_params = {'ax_lims': [600, 1250],
                    'metric': 'mean_rt',
                    'label': 'mean RT (ms)'}
        plot_scatter(self.group_stats, A_params, axes[0], self.line_ext)

        # Panel B: Switch cost scatter
        B_params = {'ax_lims': [-25, 350],
                    'metric': 'switch_cost',
                    'label': 'switch cost (ms)'}
        plot_scatter(self.group_stats, B_params, axes[1], self.line_ext)

        # Panel C: Congruency effect scatter
        C_params = {'ax_lims': [-15, 275],
                    'metric': 'con_effect',
                    'label': 'congruency effect (ms)'}
        plot_scatter(self.group_stats, C_params, axes[2], self.line_ext)

        return fig
