import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from task_dyva.visualization import PlotRTs
from task_dyva.utils import save_figure


class FigureS2():
    """Analysis methods and plotting routines to reproduce
    Figure S2 from the manuscript (example RT distributions).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    figsize = (14, 20)
    age_bin_strs = ['ages20to29', 'ages30to39', 'ages40to49', 'ages50to59', 
                'ages60to69', 'ages70to79', 'ages80to89']
    age_bin_labels= ['Ages 20 to 29', 'Ages 30 to 39', 'Ages 40 to 49', 
                     'Ages 50 to 59', 'Ages 60 to 69', 'Ages 70 to 79', 
                     'Ages 80 to 89']
    palette = 'viridis'
    
    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.all_stats = {ab: [] for ab in self.age_bin_strs}

    def make_figure(self):
        print('Making Figure S2...')
        self._run_preprocessing()
        fig = self._plot_figure()
        save_figure(fig, self.save_dir, 'FigS2')
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
            self.all_stats[ab].append(expt_stats)

    def _plot_figure(self):
        fig, axes = plt.subplots(10, 7, figsize=self.figsize)
        rt_plot_type = 'all'

        for ab_ind, ab in enumerate(self.age_bin_strs):    
            ab_stats = self.all_stats[ab]
            ab_means = np.array([s.summary_stats['u_mean_rt'] 
                                 for s in ab_stats])
            sort_inds = np.argsort(ab_means)
            plot_inds = np.arange(0, len(sort_inds), 2)
            
            for ax_ind, p in enumerate(plot_inds):
                this_ax = axes[ax_ind, ab_ind]
                this_stats = ab_stats[sort_inds[p]]
                plotter = PlotRTs(this_stats)
                this_ax = plotter.plot_rt_dists(this_ax, rt_plot_type)

                if ax_ind == 0:
                    this_ax.set_title(self.age_bin_labels[ab_ind])
                if ax_ind == 0 and ab_ind == 0:
                    this_ax.set_ylabel('RT (ms)')
                    this_ax.get_legend().set_title(None)
                    this_ax.get_legend().get_frame().set_linewidth(0.0) 
                    this_ax.legend(labels=['Participant', 'Model'])
                else:
                    this_ax.get_legend().remove()

        plt.tight_layout()
        return fig
