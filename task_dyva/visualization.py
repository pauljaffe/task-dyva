import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pdb

from .taskdataset import EbbFlowStats


class PlotRTs(EbbFlowStats):
    """Plot RT distributions.

    Args
    ----
    stats_obj (EbbFlowStats instance): data from the model/participant.
    palette (str, optional): Color palette used for plotting.
    """

    def __init__(self, stats_obj, palette='viridis'):
        self.__dict__ = stats_obj.__dict__
        self.palette = palette

    def plot_rt_dists(self, ax, plot_type):
        if plot_type == 'all':
            plot_df = self._format_all()
        elif plot_type == 'switch':
            plot_df = self._format_by_switch()
        elif plot_type == 'congruency':
            plot_df = self._format_by_congruency()

        sns.violinplot(x='trial_type', y='rts', hue='model_or_user', 
                       data=plot_df, split=True, inner=None, ax=ax,
                       palette=self.palette, cut=0, linewidth=0.5)

        if plot_type == 'all':
            ax.set_xticks([])
        else:
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha='right', 
                rotation_mode='anchor')
        ax.set_xlabel('')
        ax.set_ylabel('')
        return ax

    def _format_as_df(self, plot_dists, model_or_user, trial_types):
        all_rts = pd.concat(plot_dists)
        m_u_array = []
        ttype_array = []
        for rts, mu, ttype in zip(plot_dists, model_or_user, trial_types):
            m_u_array.extend(len(rts) * [mu])
            ttype_array.extend(len(rts) * [ttype])
        plot_df = pd.DataFrame({'rts': all_rts, 'model_or_user': m_u_array,
                                'trial_type': ttype_array})
        return plot_df

    def _format_all(self):
        plot_dists = [self.df['urt_ms'], self.df['mrt_ms']]
        m_or_u = ['user', 'model']
        trial_types = ['N/A', 'N/A']
        return self._format_as_df(plot_dists, m_or_u, trial_types)

    def _format_by_switch(self):
        stay_inds = self.select(**{'is_switch': 0})
        switch_inds = self.select(**{'is_switch': 1})
        u_stay_rts = self.df['urt_ms'][stay_inds]
        m_stay_rts = self.df['mrt_ms'][stay_inds]
        u_switch_rts = self.df['urt_ms'][switch_inds]
        m_switch_rts = self.df['mrt_ms'][switch_inds]
        plot_dists = [u_stay_rts, u_switch_rts, m_stay_rts, m_switch_rts]
        trial_types = ['Stay', 'Switch', 'Stay', 'Switch']
        m_or_u = ['user', 'user', 'model', 'model']
        return self._format_as_df(plot_dists, m_or_u, trial_types)

    def _format_by_congruency(self):
        con_inds = self.select(**{'is_congruent': 1})
        incon_inds = self.select(**{'is_congruent': 0})
        u_con_rts = self.df['urt_ms'][con_inds]
        m_con_rts = self.df['mrt_ms'][con_inds]
        u_incon_rts = self.df['urt_ms'][incon_inds]
        m_incon_rts = self.df['mrt_ms'][incon_inds]
        plot_dists = [u_con_rts, u_incon_rts, m_con_rts, m_incon_rts]
        trial_types = ['Congruent', 'Incongruent', 'Congruent',
                       'Incongruent']
        m_or_u = ['user', 'user', 'model', 'model']
        return self._format_as_df(plot_dists, m_or_u, trial_types)


class BarPlot():
    """Plot seaborn style barplots, but allow plotting of
    s.e.m. error bars. See figure2.py and figure3.py for usage.

    Args
    ----
    df (pandas DataFrame): Data to plot. 
    palette (str, optional): Color palette used for plotting.
    """

    supported_error = {'sem', 'sd'}

    def __init__(self, df, palette='viridis'):
        self.df = df
        self.palette = palette

    def plot_grouped_bar(self, x, y, hue, error_type, ax, **kwargs):
        # Note: Currently this only supports plotting two groups
        # (designated by the hue argument)
        assert error_type in self.supported_error, \
           'error_type must be one of the following: ' \
           f'{self.supported_error}'
        colors = [(0.2363, 0.3986, 0.5104, 1.0),
                  (0.2719, 0.6549, 0.4705, 1.0)]
        width = kwargs.get('width', 0.35)
        x_offset = -width / 2
        hue_types = self.df[hue].unique()
        for i, h in enumerate(hue_types):
            group_df = self.df.query(f'{hue} == @h')
            group_means, group_errors = self._get_group_data(
                group_df, x, y, error_type)
            plot_x = np.arange(len(group_means))
            ax.bar(plot_x + x_offset, group_means, yerr=group_errors,
                   width=width, label=h, **{'fc': colors[i]})
            x_offset += width
        self._adjust_bar(plot_x, ax, **kwargs)

    def plot_bar(self, keys, error_type, ax, **kwargs):
        assert error_type in self.supported_error, \
           'error_type must be one of the following: ' \
           f'{self.supported_error}'
        colors = sns.color_palette(palette=self.palette, n_colors=len(x),
                                   as_cmap=True)
        width = kwargs.get('width', 0.35)
        plot_data = [self.df[key] for key in keys]
        for di, d in enumerate(plot_data):
            d_mean = np.mean(d)
            d_sem = np.std(d) / np.sqrt(len(d))
            ax.bar(di, d_mean, d_sem, width=width, **{'fc': colors[di]})
            pdb.set_trace()
        self._adjust_bar(np.arange(len(plot_data)), ax, **kwargs)

    def _get_group_data(self, group_df, x, y, error_type):
        means = group_df.groupby(x)[y].mean().to_numpy()
        if error_type == 'sem':
            errors = group_df.groupby(x)[y].sem().to_numpy()
        elif error_type == 'sd':
            errors = group_df.groupby(x)[y].std().to_numpy()
        return means, errors

    def _adjust_bar(self, plot_x, ax, **kwargs):
        ax.set_xlabel(kwargs.get('xlabel', None))
        ax.set_ylabel(kwargs.get('ylabel', None))
        ax.set_xticks(plot_x)
        ax.set_xticklabels(kwargs.get('xticklabels', None),
                           rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticks(kwargs.get('yticks'))
        ax.set_xlim(kwargs.get('xlim', None)
        ax.set_ylim(kwargs.get('ylim', None))
        if kwargs.get('plot_legend', False):
            ax.legend()
