import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from task_dyva.visualization import BarPlot
from task_dyva.utils import save_figure, plot_scatter, expt_stats_to_df


class FigureS4():
    """Analysis methods and plotting routines to reproduce
    Figure S4 from the manuscript (RT variability).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    figsize = (9, 4.5)
    line_ext = 0.1
    age_bin_labels = ['20-29', '30-39', '40-49', '50-59', 
                      '60-69', '70-79', '80-89']

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.group_stats = {'u_rt_sd': [], 'm_rt_sd': []}
        self.analysis_expt_stats = []
        self.analysis_age_bins = []
        self.analysis_expt_strs = []

    def make_figure(self):
        print('Making Figure S4...')
        self._run_preprocessing()
        print('Stats for Figure S4')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'FigS4')
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
            self.analysis_age_bins.append(ab)
            self.analysis_expt_stats.append(expt_stats)
            self.analysis_expt_strs.append(expt_str)
            for key in self.group_stats.keys():
                self.group_stats[key].append(expt_stats.summary_stats[key])

    def _plot_figure_get_stats(self):
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Panel A: Model vs. participant scatter for RT SD
        A_params = {'ax_lims': [20, 350],
                    'metric': 'rt_sd',
                    'label': 'RT SD (ms)'}
        plot_scatter(self.group_stats, A_params, axes[0], self.line_ext)

        # Panel B: Model vs. participant RT SD binned by age
        error_type = 'sem'
        stats_df = expt_stats_to_df(['rt_sd'],
                                    self.analysis_expt_strs,
                                    self.analysis_age_bins,
                                    self.analysis_expt_stats)
        B_params = {'ylabel': 'RT SD (ms)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': True}
        B_bar = BarPlot(stats_df)
        B_bar.plot_grouped_bar('age_bin', 'value', 'model_or_user',
                               error_type, axes[1], **B_params)
        return fig
