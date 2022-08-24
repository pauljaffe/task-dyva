import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from task_dyva.utils import save_figure
from task_dyva.visualization import PlotModelLatents


class FigureS7():
    """Analysis methods and plotting routines to reproduce
    Figure S7 from the manuscript (example latent state trajectories).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    fp_fn = 'fixed_points.pkl'
    age_bins = ['ages20to29', 'ages30to39', 'ages40to49', 
                'ages50to59', 'ages60to69', 'ages70to79', 'ages80to89']
    plot_age_bins = ['ages20to29', 'ages50to59', 'ages80to89']
    plot_titles = ['Ages 20 to 29', 'Ages 50 to 59', 'Ages 80 to 89']
    figsize = (9, 13)
    figdpi = 300

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.all_stats = {ab: [] for ab in self.age_bins}
        self.all_fps = {ab: [] for ab in self.age_bins}

    def make_figure(self):
        print('Making Figure S7...')
        self._run_preprocessing()
        fig = self._plot_figure()
        save_figure(fig, self.save_dir, 'FigS7')
        print('')

    def _run_preprocessing(self):
        for expt_str, ab, sc in zip(self.expts, 
                                    self.age_bins, 
                                    self.sc_status):
            # Skip sc- models
            if sc == 'sc-':
                continue
            
            # Load stats from the holdout data
            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                expt_stats = pickle.load(path)

            # Load fixed points
            fp_path = os.path.join(self.model_dir, expt_str, 
                                   self.analysis_dir, self.fp_fn)
            with open(fp_path, 'rb') as path:
                fps = pickle.load(path)

            self.all_stats[ab].append(expt_stats)
            self.all_fps[ab].append(fps)

    def _plot_figure(self):
        fig = plt.figure(figsize=self.figsize, dpi=self.figdpi)
        nrows = 5
        t_post = 1200
        elev, azim = 30, 60
        for ab_ind, ab in enumerate(self.plot_age_bins):    
            this_stats = self.all_stats[ab]
            this_fps = self.all_fps[ab]
            this_means = np.array([s.summary_stats['u_mean_rt'] 
                                   for s in this_stats])
            sort_inds = np.argsort(this_means)
            plot_inds = np.arange(0, len(sort_inds), 20 // nrows)
            
            for ax_ind, p in enumerate(plot_inds):
                subplot_ind = ax_ind * 3 + ab_ind + 1
                ax = fig.add_subplot(nrows, 3, subplot_ind, projection='3d')
                plot_stats = this_stats[sort_inds[p]]
                plot_fps = this_fps[sort_inds[p]]

                # Plot
                if ax_ind == 0 and ab_ind == 0:
                    kwargs = {'annotate': True}
                else:
                    kwargs = {'annotate': False}
                plotter = PlotModelLatents(plot_stats, post_on_dur=t_post,
                                           fixed_points=plot_fps, plot_pre_onset=False)
                ax = plotter.plot_main_conditions(ax, elev=elev, azim=azim,
                                                  **kwargs)
                if ax_ind == 0:
                    ax.set_title(self.plot_titles[ab_ind])
        return fig
