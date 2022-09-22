import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from task_dyva.utils import save_figure
from task_dyva.visualization import PlotModelLatents


class Figure3():
    """Analysis methods and plotting routines to reproduce
    Figure 3 from the manuscript (hierarchical task representation).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'holdout_outputs_01SD.pkl'
    fp_fn = 'fixed_points.pkl'
    fp_summary_fn = 'fixed_point_summary.pkl'
    distance_keys = ['within_task', 'between_task', 
                     'same_response', 'different_response']
    lda_summary_fn = 'lda_summary.pkl'
    example_user = 3139
    figsize = (5.5, 4)
    figdpi = 300
    kde_colors = sns.color_palette(palette='viridis', n_colors=4)

    def __init__(self, model_dir, save_dir, metadata):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.expts = metadata['name']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']
        self.exgauss = metadata['exgauss']
        self.early = metadata['early']
        self.optimal = metadata['optimal']

        # Containers for summary stats
        self.group_pca_summary = []
        self.group_fp_summary = []
        self.group_lda_summary = []
        self.ex_fps = None
        self.ex_stats = None

    def make_figure(self):
        print('Making Figure 3...')
        self._run_preprocessing()
        print('Stats for Figure 3')
        print('------------------')
        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, 'Fig3')
        print('')

    def _run_preprocessing(self):
        for expt_str, uid, sc, exg, early, opt in zip(self.expts, 
                                                      self.user_ids, 
                                                      self.sc_status,
                                                      self.exgauss,
                                                      self.early,
                                                      self.optimal):

            # Skip sc- models, exgauss+ models, early models, optimal models
            if sc == 'sc-' or exg == 'exgauss+' or early or opt:
                continue
            
            # Load stats from the holdout data
            stats_path = os.path.join(self.model_dir, expt_str, 
                                      self.analysis_dir, self.stats_fn)
            with open(stats_path, 'rb') as path:
                expt_stats = pickle.load(path)

            # Variance explained vs. PC number
            self.group_pca_summary.append(expt_stats.pca_explained_var)

            # Fixed points
            if uid == self.example_user:
                fp_path = os.path.join(self.model_dir, expt_str, 
                                       self.analysis_dir, self.fp_fn)
                with open(fp_path, 'rb') as path:
                    self.ex_fps = pickle.load(path)
                self.ex_stats = expt_stats

            fp_summary_path = os.path.join(self.model_dir, expt_str, 
                                           self.analysis_dir, self.fp_summary_fn)
            with open(fp_summary_path, 'rb') as path:
                fp_summary = pickle.load(path)
            self.group_fp_summary.append(
                self._get_user_fp_stats(fp_summary, expt_str))

            # LDA analyses
            lda_summary_path = os.path.join(self.model_dir, expt_str, 
                                            self.analysis_dir, self.lda_summary_fn)
            with open(lda_summary_path, 'rb') as path:
                lda_summary = pickle.load(path)
            lda_summary['expt'] = expt_str
            self.group_lda_summary.append(pd.DataFrame(lda_summary, index=[0]))

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize, 
                         dpi=self.figdpi)
        gs = fig.add_gridspec(11, 20)

        # Panel A: Example model trajectories + fixed points
        axA = fig.add_subplot(gs[0:11, 0:11], projection='3d')
        self._make_panel_A(axA)

        # Panel B: PC number vs. variance explained
        axB = fig.add_subplot(gs[0:2, 13:20])
        self._make_panel_B(axB)

        # Panel C: LDA summary
        axC = fig.add_subplot(gs[3:5, 13:20])
        self._make_panel_C(axC)

        # Get summary statistics for the fixed points
        group_fp_df = self._get_group_fp_stats()

        # Panel D: Distance between fixed points
        axD = fig.add_subplot(gs[6:8, 13:20])
        self._make_panel_D(axD, group_fp_df)

        return fig

    def _make_panel_A(self, ax):
        # Plotting params
        t_post = 1200
        elev, azim = 30, 60
        kwargs = {'xlim': [-25, 28], 'ylim': [-20, 10], 'zlim': [-10, 7]}
        # Plot
        plotter = PlotModelLatents(self.ex_stats, post_on_dur=t_post, 
                                   fixed_points=self.ex_fps, plot_pre_onset=False)
        _ = plotter.plot_main_conditions(ax, elev=elev, azim=azim, 
                                         plot_task_centroid=False, **kwargs)

    def _make_panel_B(self, ax):
        # Note error bars show the SD
        n_pcs = 5
        yticks = [0, 0.25, 0.5, 0.75, 1]
        ylim = [0, 1.1]

        data = self.group_pca_summary
        data_csum = np.cumsum(np.stack(data, axis=0), axis=1)
        data_mean = np.mean(data_csum, axis=0)
        data_sd = np.std(data_csum, axis=0)
        data_sem = data_sd / np.sqrt(len(data))
        x = (np.arange(n_pcs) + 1).astype('int')
        for d in range(len(data)):
            ax.plot(x, data_csum[d, :n_pcs], c='0.8', alpha=0.5, 
                    zorder=1, linewidth=0.5)
        ax.errorbar(x, data_mean[:n_pcs], yerr=data_sd[:n_pcs], capsize=0, 
                    c='k', zorder=2, linewidth=0.5)
        ax.set_xlabel('PC #')
        ax.set_ylabel('Cumulative\nexplained variance')
        ax.set_xlim([0.75, n_pcs + 0.25])
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_ylim(ylim)
        
        # Stats
        for pc_ind, pc in enumerate(x):
            print(f'PC {pc} mean +/- s.e.m. cumulative explained var.: ' \
                  f'{data_mean[pc_ind]} +/- {data_sem[pc_ind]}')

    def _make_panel_C(self, ax):
        df = pd.concat(self.group_lda_summary, ignore_index=True)
        keys = ['bw_error', 'bw_shuffle_error', 
                'within_error', 'within_shuffle_error']
        df_plot = pd.melt(df, id_vars=['expt'], value_vars=keys)
        plot_labels = ['Between task', 'Between task shuffle', 
                       'Within task', 'Within task shuffle']
        ylabel = 'Misclassification rate'
        yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        xlim = [-0.75, 3.75]
        ylim = [0, 0.55]
        error_type = 'sem'
        kwargs = {'xticklabels': plot_labels, 'ylabel': ylabel,
                  'yticks': yticks, 'xlim': xlim, 'ylim': ylim}
        sns.kdeplot(data=df_plot, x='value', hue='variable', ax=ax,
                    common_norm=False, cumulative=True, linewidth=0.5)
        ax.set_xlabel('Misclassification rate')
        ax.set_ylabel('Density')
        ax.get_legend().set_title(None)
        
        # Stats
        print('LDA analysis stats:')
        w_task, p_task = wilcoxon(df['bw_error'].values, 
                                  y=df['bw_shuffle_error'].values,
                                  mode='approx')
        w_direction, p_direction = wilcoxon(df['within_error'].values, 
                                            y=df['within_shuffle_error'].values,
                                            mode='approx')
        print(f'Within vs. between task, signed-rank test: w_stat = {w_task}, ' \
              f'p = {p_task}, N = {len(df)}')
        print('Within task, same vs. different relevant direction, ' \
              f'signed-rank: w_stat = {w_direction}, p = {p_direction}, ' \
              f'N = {len(df)}')
        for key in keys:
            print(f'{key} mean +/- s.e.m. misclassification rate: ' \
                  f'{df[key].mean()} +/- {df[key].sem()}')
        print('---------------------------------------')

    def _make_panel_D(self, ax, df):
        keys = ['within_task', 'between_task', 
                'same_response', 'different_response']
        df_plot = pd.melt(df, id_vars=['expt'], value_vars=keys)
        error_type = 'sem'
        plot_labels = ['Within task', 'Between task', 
                       'Same direction', 'Different direction']
        ylabel = 'Euclidean distance\nbetween fixed points (a.u.)'
        yticks = np.arange(0, 35, 5).astype('int')
        xlim = [-0.75, 3.75]
        ylim = [0, 30]
        kwargs = {'xticklabels': plot_labels, 'ylabel': ylabel,
                  'yticks': yticks, 'xlim': xlim, 'ylim': ylim}
        sns.kdeplot(data=df_plot, x='value', hue='variable', ax=ax,
                    common_norm=False, cumulative=True, linewidth=0.5)
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Density')
        ax.get_legend().set_title(None)

        # Stats
        print('Stats on distance between fixed points:')
        w_task, p_task = wilcoxon(df['within_task'].values, 
                                  y=df['between_task'].values,
                                  mode='approx')
        w_direction, p_direction = wilcoxon(df['same_response'].values, 
                                            y=df['different_response'].values,
                                            mode='approx')
        print(f'Within vs. between task, signed-rank test: w_stat = {w_task}, ' \
              f'p = {p_task}, N = {len(df)}')
        print('Within task, same vs. different relevant direction, ' \
              f'signed-rank: w_stat = {w_direction}, p = {p_direction}, ' \
              f'N = {len(df)}')

    def _get_user_fp_stats(self, data, expt_str):
        stats = {}
        for key in self.distance_keys:
            if len(data[key]) == 0:
                stats[key] = np.nan
            else:
                stats[key] = np.mean(data[key])
        stats['expt'] = expt_str
        stats['N'] = data['N']
        stats['f_stimuli_with_fp'] = data['f_stimuli_with_fp']
        return pd.DataFrame(stats, index=[0])

    def _get_group_fp_stats(self):
        print('Stats on number of fixed points, all models:')
        df = pd.concat(self.group_fp_summary, ignore_index=True)
        N_zero = len(df.query('N == 0'))
        print(f'N models with no fixed points: {N_zero}')
        print('--------------------------------------------')

        # Check counts for each of the distance keys with np.isnan
        for key in self.distance_keys:
            N_na = df[key].isna().sum()
            print(f'N models with no pairs for {key}: {N_na}')

        # Summary stats for models included in distance analyses
        df_filt = df.dropna(axis=0, how='any')
        N10 = len(df_filt.query('N >= 10'))
        N_mean = df_filt['N'].mean()
        N_sem = df_filt['N'].sem()
        f_mean = df_filt['f_stimuli_with_fp'].mean()
        f_sem = df_filt['f_stimuli_with_fp'].sem()
        print('Stats on fixed points, models included in distance analyses:')
        print(f'N models included in distance analyses: {len(df_filt)}') 
        print(f'N models with at least ten fixed points: {N10}')
        print(f'Mean +/- s.e.m. fixed points per model: {N_mean} +/- {N_sem}')
        print('Mean +/- s.e.m. fraction of possible stimulus configurations' \
              f'with a fixed point: {f_mean} +/- {f_sem}')
        print('------------------------------------------------------------')
        return df_filt
