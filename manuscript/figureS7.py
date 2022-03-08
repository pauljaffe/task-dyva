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
    Figure S7 from the manuscript (dynamical origins of the switch cost;
    example trajectories). 
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    figsize = (10, 2.5)
    figdpi = 300
    palette = 'viridis'

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']

        # Containers for summary stats
        self.ex = {4913: {'mv': 0, 'pt': 0, 'cue': 1},
                   3538: {'mv': 1, 'pt': 1, 'cue': 0},
                   3531: {'mv': 2, 'pt': 2, 'cue': 1},
                   1276: {'mv': 2, 'pt': 2, 'cue': 0}}

    def make_figure(self):
        print('Making Figure S7...')
        self._run_preprocessing()
        print('------------------')
        fig = self._plot_figure()
        save_figure(fig, self.save_dir, 'FigS7_trajectories')
        print('')

    def _run_preprocessing(self):
        for expt_str, uid, sc in zip(self.expts, 
                                     self.user_ids, 
                                     self.sc_status):

            if uid in self.ex.keys() and sc != 'sc-':
                # Load stats from the holdout data
                stats_path = os.path.join(self.model_dir, expt_str, 
                                          self.analysis_dir, self.stats_fn)
                with open(stats_path, 'rb') as path:
                    expt_stats = pickle.load(path)
                self.ex[uid]['stats'] = expt_stats

    def _plot_figure(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(3, 16)

        # Panel C: Example model latent state trajectories
        for i, uid in enumerate(self.ex.keys()):
            ax = fig.add_subplot(gs[0:3, i*4:i*4+3], projection='3d')
            params = self.ex[uid]
            _ = self._make_ex_panel(ax, params)

        return fig

    def _make_ex_panel(self, ax, params):
        t_post = 1000
        elev, azim = 30, 45
        stats = params.pop('stats')
        plotter = PlotModelLatents(stats, post_on_dur=t_post,
                                   plot_pre_onset=False)
        _ = plotter.plot_stay_switch(ax, params, elev=elev, azim=azim)
        return ax
