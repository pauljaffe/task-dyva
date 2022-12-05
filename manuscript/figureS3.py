import os
import pickle
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.anova import AnovaRM

from task_dyva.utils import save_figure, plot_scatter, expt_stats_to_df, adjust_boxplot


class FigureS3():
    """Analysis methods and plotting routines to reproduce
    Figure S3 from the manuscript (model vs. participant accuracy;
    model vs. participant RT variability; interaction b/w switch and congruency).
    """

    analysis_dir = 'model_analysis'
    stats_fn = 'summary.pkl'
    outputs_fn = 'holdout_outputs_01SD.pkl'
    figsize = (7, 3)
    figdpi = 300
    box_colors = {'Participants': (0.2363, 0.3986, 0.5104, 1.0),
                  'Models': (0.2719, 0.6549, 0.4705, 1.0)}

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
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
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
        self.inter_rt_dfs = []
        self.inter_acc_dfs = []

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

            # Congruency/switch interaction stats
            inter_rts, inter_acc = self._get_interaction_stats(var_expt_stats, expt_str)
            self.inter_rt_dfs.append(inter_rts)
            self.inter_acc_dfs.append(inter_acc)

        self.inter_rt_df = pd.concat(self.inter_rt_dfs)
        self.inter_acc_df = pd.concat(self.inter_acc_dfs)
        self.inter_rt_df.reset_index(drop=True, inplace=True)
        self.inter_acc_df.reset_index(drop=True, inplace=True)

    def _get_interaction_stats(self, data, expt_str):
        # Stay RTs
        u_rt_stay_con_inds = data.select(**{'is_switch': 0, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 1})
        u_rt_stay_con = data.df['urt_ms'][u_rt_stay_con_inds].mean() 
        m_rt_stay_con_inds = data.select(**{'is_switch': 0, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 1})
        m_rt_stay_con = data.df['mrt_ms'][m_rt_stay_con_inds].mean() 
        u_rt_stay_incon_inds = data.select(**{'is_switch': 0, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 0})
        u_rt_stay_incon = data.df['urt_ms'][u_rt_stay_incon_inds].mean() 
        m_rt_stay_incon_inds = data.select(**{'is_switch': 0, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 0})
        m_rt_stay_incon = data.df['mrt_ms'][m_rt_stay_incon_inds].mean() 
        # Switch RTs
        u_rt_switch_con_inds = data.select(**{'is_switch': 1, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 1})
        u_rt_switch_con = data.df['urt_ms'][u_rt_switch_con_inds].mean() 
        m_rt_switch_con_inds = data.select(**{'is_switch': 1, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 1})
        m_rt_switch_con = data.df['mrt_ms'][m_rt_switch_con_inds].mean() 
        u_rt_switch_incon_inds = data.select(**{'is_switch': 1, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 0})
        u_rt_switch_incon = data.df['urt_ms'][u_rt_switch_incon_inds].mean() 
        m_rt_switch_incon_inds = data.select(**{'is_switch': 1, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 0})
        m_rt_switch_incon = data.df['mrt_ms'][m_rt_switch_incon_inds].mean() 
        rt_df = pd.DataFrame(np.array(
                [[u_rt_stay_con, 'Congruent', 'Stay', 'Participants', 'congruent/stay'],
                 [m_rt_stay_con, 'Congruent', 'Stay', 'Models', 'congruent/stay'],
                 [u_rt_stay_incon, 'Incongruent', 'Stay', 'Participants', 'incongruent/stay'],
                 [m_rt_stay_incon, 'Incongruent', 'Stay', 'Models', 'incongruent/stay'],
                 [u_rt_switch_con, 'Congruent', 'Switch', 'Participants', 'congruent/switch'],
                 [m_rt_switch_con, 'Congruent', 'Switch', 'Models', 'congruent/switch'],
                 [u_rt_switch_incon, 'Incongruent', 'Switch', 'Participants', 'incongruent/switch'],
                 [m_rt_switch_incon, 'Incongruent', 'Switch', 'Models', 'incongruent/switch']]),
            columns = ['rt', 'con_incon', 'stay_switch', 'model_user', 'condition'])
        rt_df['name'] = 8 * [expt_str]

        # Stay accuracy
        u_acc_stay_con_correct_inds = data.select(**{'is_switch': 0, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 1})
        u_acc_stay_con_incorrect_inds = data.select(**{'is_switch': 0, 'ucorrect': 0, 
            'u_prev_correct': 1, 'is_congruent': 1})
        u_acc_stay_con = self._get_condition_acc(u_acc_stay_con_correct_inds,
                                                 u_acc_stay_con_incorrect_inds,
                                                 data.df)
        m_acc_stay_con_correct_inds = data.select(**{'is_switch': 0, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 1})
        m_acc_stay_con_incorrect_inds = data.select(**{'is_switch': 0, 'mcorrect': 0, 
            'm_prev_correct': 1, 'is_congruent': 1})
        m_acc_stay_con = self._get_condition_acc(m_acc_stay_con_correct_inds,
                                                 m_acc_stay_con_incorrect_inds,
                                                 data.df)
        u_acc_stay_incon_correct_inds = data.select(**{'is_switch': 0, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 0})
        u_acc_stay_incon_incorrect_inds = data.select(**{'is_switch': 0, 'ucorrect': 0, 
            'u_prev_correct': 1, 'is_congruent': 0})
        u_acc_stay_incon = self._get_condition_acc(u_acc_stay_incon_correct_inds,
                                                   u_acc_stay_incon_incorrect_inds,
                                                   data.df)
        m_acc_stay_incon_correct_inds = data.select(**{'is_switch': 0, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 0})
        m_acc_stay_incon_incorrect_inds = data.select(**{'is_switch': 0, 'mcorrect': 0, 
            'm_prev_correct': 1, 'is_congruent': 0})
        m_acc_stay_incon = self._get_condition_acc(m_acc_stay_incon_correct_inds,
                                                   m_acc_stay_incon_incorrect_inds,
                                                   data.df)
        # Switch accuracy
        u_acc_switch_con_correct_inds = data.select(**{'is_switch': 1, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 1})
        u_acc_switch_con_incorrect_inds = data.select(**{'is_switch': 1, 'ucorrect': 0, 
            'u_prev_correct': 1, 'is_congruent': 1})
        u_acc_switch_con = self._get_condition_acc(u_acc_switch_con_correct_inds,
                                                   u_acc_switch_con_incorrect_inds,
                                                   data.df)
        m_acc_switch_con_correct_inds = data.select(**{'is_switch': 1, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 1})
        m_acc_switch_con_incorrect_inds = data.select(**{'is_switch': 1, 'mcorrect': 0, 
            'm_prev_correct': 1, 'is_congruent': 1})
        m_acc_switch_con = self._get_condition_acc(m_acc_switch_con_correct_inds,
                                                   m_acc_switch_con_incorrect_inds,
                                                   data.df)
        u_acc_switch_incon_correct_inds = data.select(**{'is_switch': 1, 'ucorrect': 1, 
            'u_prev_correct': 1, 'is_congruent': 0})
        u_acc_switch_incon_incorrect_inds = data.select(**{'is_switch': 1, 'ucorrect': 0, 
            'u_prev_correct': 1, 'is_congruent': 0})
        u_acc_switch_incon = self._get_condition_acc(u_acc_switch_incon_correct_inds,
                                                     u_acc_switch_incon_incorrect_inds,
                                                     data.df)
        m_acc_switch_incon_correct_inds = data.select(**{'is_switch': 1, 'mcorrect': 1, 
            'm_prev_correct': 1, 'is_congruent': 0})
        m_acc_switch_incon_incorrect_inds = data.select(**{'is_switch': 1, 'mcorrect': 0, 
            'm_prev_correct': 1, 'is_congruent': 0})
        m_acc_switch_incon = self._get_condition_acc(m_acc_switch_incon_correct_inds,
                                                     m_acc_switch_incon_incorrect_inds,
                                                     data.df)
        acc_df = pd.DataFrame(np.array(
                [[u_acc_stay_con, 'Congruent', 'Stay', 'Participants', 'congruent/stay'],
                 [m_acc_stay_con, 'Congruent', 'Stay', 'Models', 'congruent/stay'],
                 [u_acc_stay_incon, 'Incongruent', 'Stay', 'Participants', 'incongruent/stay'],
                 [m_acc_stay_incon, 'Incongruent', 'Stay', 'Models', 'incongruent/stay'],
                 [u_acc_switch_con, 'Congruent', 'Switch', 'Participants', 'congruent/switch'],
                 [m_acc_switch_con, 'Congruent', 'Switch', 'Models', 'congruent/switch'],
                 [u_acc_switch_incon, 'Incongruent', 'Switch', 'Participants', 'incongruent/switch'],
                 [m_acc_switch_incon, 'Incongruent', 'Switch', 'Models', 'incongruent/switch']]),
            columns = ['acc', 'con_incon', 'stay_switch', 'model_user', 'condition'])
        acc_df['name'] = 8 * [expt_str]
        return rt_df, acc_df

    def _get_condition_acc(self, correct, incorrect, df):
        n_correct = len(df.iloc[correct, :])
        n_incorrect = len(df.iloc[incorrect, :])
        return n_correct / (n_correct + n_incorrect)

    def _plot_figure_get_stats(self):
        fig = plt.figure(constrained_layout=False, figsize=self.figsize,
                         dpi=self.figdpi)
        gs = fig.add_gridspec(15, 28)

        # Accuracy panels: a-c
        A_ax = fig.add_subplot(gs[0:6, 0:8])
        B_ax = fig.add_subplot(gs[0:6, 10:18])
        C_ax = fig.add_subplot(gs[0:6, 20:28])
        acc_axes = [A_ax, B_ax, C_ax]
        for key, label, ax in zip(self.plot_keys, self.plot_labels, acc_axes):
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
            plt.setp(ax.get_xticklabels()[1::2], visible=False)
            ax.set_ylabel(label)
            if key == 'accuracy':
                ax.legend()
                ax.get_legend().get_frame().set_linewidth(0.0)   
            if key == 'acc_con_effect':
                ax.set_xlabel('Noise SD')
            else:
                ax.set_xlabel('')

        # Panel d: Model vs. participant scatter for RT SD
        D_params = {'ax_lims': [15, 350],
                    'metric': 'rt_sd',
                    'label': 'RT SD (ms)'}
        D_ax = fig.add_subplot(gs[9:15, 0:5])
        plot_scatter(self.var_stats, D_params, D_ax, self.line_ext,
                     self.rng, n_boot=self.n_boot, alpha=self.alpha)

        # Panel e: Model vs. participant RT SD binned by age
        error_type = 'sem'
        stats_df = expt_stats_to_df(['rt_sd'],
                                    self.analysis_expt_strs,
                                    self.analysis_age_bins,
                                    self.analysis_expt_stats)
        E_params = {'ylabel': 'RT SD (ms)',
                    'xlabel': 'Age bin (years)',
                    'xticklabels': self.age_bin_labels,
                    'plot_legend': True}
        E_ax = fig.add_subplot(gs[9:15, 7:14])
        sns.boxplot(data=stats_df, x='age_bin', y='value', hue='model_or_user',
                    ax=E_ax, orient='v', fliersize=1, palette=self.box_colors,
                    linewidth=0.5)
        adjust_boxplot(E_ax, **E_params)
        print('------------------------------')

        # Panels f-g: Interaction b/w switch/congruency for RTs (f) and accuracy (g)
        print('RT congruency/switch interaction stats:')
        print('------------------------------')
        F_ax = fig.add_subplot(gs[9:15, 16:21])
        self._make_inter_panel(self.inter_rt_df, F_ax, 'rt')
        F_params = {'ylabel': 'Mean RT (ms)',
                    'plot_legend': True}
        adjust_boxplot(F_ax, **F_params)

        print('Accuracy congruency/switch interaction stats:')
        G_ax = fig.add_subplot(gs[9:15, 23:28])
        self._make_inter_panel(self.inter_acc_df, G_ax, 'acc')
        G_params = {'ylabel': 'Accuracy',
                    'plot_legend': True}
        adjust_boxplot(G_ax, **G_params)

        return fig

    def _make_inter_panel(self, df, ax, key):
        df[key] = pd.to_numeric(df[key])
        # Stats
        print('Model stats:')
        m_df = df.query("model_user == 'Models'")
        m_df.reset_index(drop=True, inplace=True)
        print(AnovaRM(data=m_df, depvar=key, subject='name', 
                      within=['con_incon', 'stay_switch']).fit())

        print('Participant stats:')
        u_df = df.query("model_user == 'Participants'")
        u_df.reset_index(drop=True, inplace=True)
        print(AnovaRM(data=u_df, depvar=key, subject='name', 
                      within=['con_incon', 'stay_switch']).fit())

        # Plot
        sns.boxplot(data=df, x='condition', y=key, hue='model_user',
                    ax=ax, orient='v', fliersize=1, linewidth=0.5,
                    palette=self.box_colors)

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
            print(f'Participant vs. model signed-rank p-val: {p}, W = {w}')
            print(f'Participant mean +/- s.e.m.: {u_mean} +/- {u_sem}')
            print(f'Model mean +/- s.e.m.: {m_mean} +/- {m_sem}')
            print('----------------------------------')
        return u_mean, m_mean, u_sem, m_sem
