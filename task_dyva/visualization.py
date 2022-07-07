import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .taskdataset import EbbFlowStats


class PlotRTs(EbbFlowStats):
    """Plot RT distributions.

    Args
    ----
    stats_obj (EbbFlowStats instance): Data from the model/participant.
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
        urts = self.df['urt_ms'][self.select(**{'ucorrect': 1})]
        mrts = self.df['mrt_ms'][self.select(**{'mcorrect': 1})]
        plot_dists = [urts, mrts]
        m_or_u = ['user', 'model']
        trial_types = ['N/A', 'N/A']
        return self._format_as_df(plot_dists, m_or_u, trial_types)

    def _format_by_switch(self):
        u_stay_inds = self.select(**{'is_switch': 0, 'ucorrect': 1, 
                                     'u_prev_correct': 1})
        m_stay_inds = self.select(**{'is_switch': 0, 'mcorrect': 1, 
                                     'm_prev_correct': 1})
        u_switch_inds = self.select(**{'is_switch': 1, 'ucorrect': 1, 
                                       'u_prev_correct': 1})
        m_switch_inds = self.select(**{'is_switch': 1, 'mcorrect': 1, 
                                       'm_prev_correct': 1})
        u_stay_rts = self.df['urt_ms'][u_stay_inds]
        m_stay_rts = self.df['mrt_ms'][m_stay_inds]
        u_switch_rts = self.df['urt_ms'][u_switch_inds]
        m_switch_rts = self.df['mrt_ms'][m_switch_inds]
        plot_dists = [u_stay_rts, u_switch_rts, m_stay_rts, m_switch_rts]
        trial_types = ['Stay', 'Switch', 'Stay', 'Switch']
        m_or_u = ['user', 'user', 'model', 'model']
        return self._format_as_df(plot_dists, m_or_u, trial_types)

    def _format_by_congruency(self):
        u_con_inds = self.select(**{'is_congruent': 1, 'ucorrect': 1})
        m_con_inds = self.select(**{'is_congruent': 1, 'mcorrect': 1})
        u_incon_inds = self.select(**{'is_congruent': 0, 'ucorrect': 1})
        m_incon_inds = self.select(**{'is_congruent': 0, 'mcorrect': 1})
        u_con_rts = self.df['urt_ms'][u_con_inds]
        m_con_rts = self.df['mrt_ms'][m_con_inds]
        u_incon_rts = self.df['urt_ms'][u_incon_inds]
        m_incon_rts = self.df['mrt_ms'][m_incon_inds]
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
        elinewidth = kwargs.get('elinewidth', 0.5)
        error_kw = {'elinewidth': elinewidth}
        for i, h in enumerate(hue_types):
            group_df = self.df.query(f'{hue} == @h')
            group_means, group_errors = self._get_group_data(
                group_df, x, y, error_type)
            plot_x = np.arange(len(group_means))
            ax.bar(plot_x + x_offset, group_means, yerr=group_errors,
                   width=width, label=h, error_kw=error_kw, 
                   **{'fc': colors[i]})
            x_offset += width
        ax = self._adjust_bar(plot_x, ax, **kwargs)
        return ax

    def plot_bar(self, keys, error_type, ax, **kwargs):
        assert error_type in self.supported_error, \
           'error_type must be one of the following: ' \
           f'{self.supported_error}'
        colors = sns.color_palette(palette=self.palette, n_colors=len(keys))
        width = kwargs.get('width', 0.75)
        plot_data = [self.df[key] for key in keys]
        for di, d in enumerate(plot_data):
            d_mean = np.mean(d)
            d_sem = np.std(d) / np.sqrt(len(d))
            ax.bar(di, d_mean, yerr=d_sem, width=width, error_kw={'elinewidth': 1},
                   **{'fc': colors[di]})
        ax = self._adjust_bar(np.arange(len(plot_data)), ax, **kwargs)
        return ax

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
        if 'yticks' in kwargs.keys():
            ax.set_yticks(kwargs['yticks'])
        ax.set_xlim(kwargs.get('xlim', None))
        ax.set_ylim(kwargs.get('ylim', None))
        if kwargs.get('plot_legend', False):
            ax.legend()
            ax.get_legend().get_frame().set_linewidth(0.0)  
        return ax


class PlotModelLatents():
    """Plot the model latents in 3D. See e.g. figure3.py and 
    figure4.py for usage.

    Args
    ----
    data (EbbFlowStats instance): Data to plot.
    post_on_dur (int, optional): Duration after stimulus onset to plot (ms).
    pcs_to_plot (list, optional): Which PCs to plot. 
    fixed_points (pandas DataFrame, optional): Fixed points to plot. 
    """
    
    default_colors = 2 * ['royalblue', 'forestgreen', 'crimson', 'orange']

    def __init__(self, data, post_on_dur=1200, pcs_to_plot=[0, 1, 2],
                 fixed_points=None, plot_pre_onset=True):
        self.data = data
        self.pcs_to_plot = pcs_to_plot
        self.latents = data.windowed['pca_latents'][:, :, pcs_to_plot]
        self.m_rts = data.df['mrt_ms'].to_numpy()
        self.step = data.step
        self.n_pre = data.n_pre
        self.t_off_ind = self.n_pre + 1 \
                         + np.round(post_on_dur / self.step).astype('int')
        if plot_pre_onset:
            self.t_on_ind = 0
        else:
            self.t_on_ind = self.n_pre
        self.fixed_points = fixed_points

    def plot_stay_switch(self, ax, params, elev=30, azim=60):
        # Plot average stay and switch trajectories for one task cue
        # and one response direction (used e.g. in Fig. 4A).
        labels = ['Stay', 'Switch']
        styles = ['-', '-']
        series = self._get_stay_switch_series(params)
        cmap = sns.color_palette('viridis', as_cmap=True)
        color_indices = [0.2, 0.8]
        colors = [cmap(i) for i in color_indices]
        plot_kwargs = {'colors': colors, 'line_styles': styles,
                       'line_width': 0.5, 'plot_series_onset': True,
                       'plot_series_rt': True, 'plot_times': np.array([100])}
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, 
                          **plot_kwargs)
        return ax

    def _get_stay_switch_series(self, params):
        pt, mv, cue = (params['pt'], params['mv'], params['cue'])
        stay_filter = {'point_dir': pt, 'mv_dir': mv, 'task_cue': cue,
                       'prev_task_cue': cue}
        switch_filter = {'point_dir': pt, 'mv_dir': mv, 'task_cue': cue,
                         'prev_task_cue': 1 - cue}
        stay_inds = self.data.select(**stay_filter)
        switch_inds = self.data.select(**switch_filter)
        all_selections = [stay_inds, switch_inds]
        return all_selections

    def plot_main_conditions(self, ax, elev=30, azim=60, 
                             plot_task_centroid=False, **kwargs):
        # Plot the 8 task cue x relevant stimulus direction combinations;
        # also plot the fixed points. Used e.g. in Figs. 3A and S6. 
        stim_cue_vals = [(0, 0),
                         (0, 1),
                         (0, 2),
                         (0, 3),
                         (1, 0),
                         (1, 1),
                         (1, 2),
                         (1, 3)]
        labels = ['Moving L', 
                  'Moving R', 
                  'Moving U',
                  'Moving D',
                  'Pointing L', 
                  'Pointing R',
                  'Pointing U',
                  'Pointing D']
        styles = ['-', '-', '-', '-',
                  '--', '--', '--', '--']
        series = self._get_main_series(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 
                       'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, 
                          **plot_kwargs)
        return ax

    def _get_main_series(self, stim_cue_vals):
        all_selections = []
        for this_stim_cue in stim_cue_vals:
            this_cue = this_stim_cue[0]
            this_stim = this_stim_cue[1]
            if this_cue == 0: # moving task
                this_filters = {'mv_dir': this_stim,                   
                                'task_cue': this_cue}
            else: # pointing task
                this_filters = {'point_dir': this_stim,                   
                                'task_cue': this_cue}    
            this_inds = self.data.select(**this_filters)
            all_selections.append(this_inds)
        return all_selections

    def plot_3d(self, series, labels, ax, elev=30, azim=60, **kwargs):
        # series should be a list of numpy arrays: the model latents
        # are averaged over each array of indices, then plotted. 
        colors = kwargs.get('colors', self.default_colors)
        line_styles = kwargs.get('line_styles', len(series) * ['-']) 
        width = kwargs.get('line_width', 0.5)
        rt_marker = kwargs.get('rt_marker', 'o')
        if kwargs.get('plot_task_centroid', False):
            ax = self._plot_task_centroid(ax)
        plot_series_onset = kwargs.get('plot_series_onset', False)
        plot_series_rt = kwargs.get('plot_series_rt', False)
        plot_times = kwargs.get('plot_times', None)

        for i, s in enumerate(series):
            label = labels[i]
            color = colors[i]
            style = line_styles[i]
            ax = self._plot_3d_line(ax, s, color, style, label, width)
            if plot_series_onset:
                ax = self._mark_3d_plot(ax, s, self.n_pre, 'k', 10, '.')
            if plot_series_rt:
                t_ind = self._get_series_rt_samples(s) + self.n_pre
                ax = self._mark_3d_plot(ax, s, t_ind, color, 6, 'o')
            if plot_times is not None:
                ax = self._plot_timepoints(ax, s, plot_times, color)

        if self.fixed_points is not None:
            ax = self._plot_fixed_points(ax)
        ax = self._adjust_plot(ax, elev, azim, **kwargs)
        return ax

    def _plot_timepoints(self, ax, series, times, color):
        for t in times:
            t_plot = int(self.n_pre + t / self.step)
            ax = self._mark_3d_plot(ax, series, t_plot, color, 7, 'x')
        return ax

    def _get_series_rt_samples(self, series):
        rt = np.round(np.mean(self.m_rts[series]) / self.step).astype('int')
        return rt

    def _plot_task_centroid(self, ax):
        color, size, marker = 'k', 20, '*'
        t_ind = self.n_pre
        mv_filter = {'task_cue': 0, 'prev_task_cue': 0}
        mv_inds = self.data.select(**mv_filter)
        pt_filter = {'task_cue': 1, 'prev_task_cue': 1}
        pt_inds = self.data.select(**pt_filter)
        for inds in [mv_inds, pt_inds]:
            ax = self._mark_3d_plot(ax, inds, t_ind, color, size, marker)
        return ax

    def _mark_3d_plot(self, ax, series_inds, t_ind, color, size, marker):
        x = np.mean(self.latents[:, series_inds, 0], 1)
        y = np.mean(self.latents[:, series_inds, 1], 1)
        z = np.mean(self.latents[:, series_inds, 2], 1)
        ax.scatter(x[t_ind], y[t_ind], z[t_ind], marker=marker,
                   facecolors=color, s=size,
                   linewidth=0.5, label=None)
        return ax

    def _plot_3d_line(self, ax, series_inds, color, style, label, width):
        x = np.mean(self.latents[self.t_on_ind:self.t_off_ind, series_inds, 0], 1)
        y = np.mean(self.latents[self.t_on_ind:self.t_off_ind, series_inds, 1], 1)
        z = np.mean(self.latents[self.t_on_ind:self.t_off_ind, series_inds, 2], 1)
        ax.plot(x, y, z, color=color, label=label, linestyle=style, 
                linewidth=width)
        return ax

    def _plot_fixed_points(self, ax):
        mv_fps = self.fixed_points.query('cue == 0')
        pt_fps = self.fixed_points.query('cue == 1')
        plot_fps = [mv_fps, pt_fps]
        fp_markers = ['x', 'x']
        colors = ['crimson', 'royalblue']
        size = 10
        for plot_fp, mark, c in zip(plot_fps, fp_markers, colors):
            for fpz in plot_fp['zloc_pca']:
                ax.scatter(fpz[0], fpz[1], fpz[2], s=size, color=c, 
                           marker=mark, zorder=2, linewidth=0.5)
        return ax

    def _adjust_plot(self, ax, elev, azim, **kwargs):
        ax.view_init(elev=elev, azim=azim)
        ax.xaxis._axinfo['grid']['linewidth'] = 0.25
        ax.yaxis._axinfo['grid']['linewidth'] = 0.25
        ax.zaxis._axinfo['grid']['linewidth'] = 0.25
        ax.set_xlim(kwargs.get('xlim', None))
        ax.set_ylim(kwargs.get('ylim', None))
        ax.set_zlim(kwargs.get('zlim', None))
        if kwargs.get('remove_tick_labels', True):
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.set_zticklabels('')
        if kwargs.get('annotate', True):
            ax = self._annotate_plot(ax, **kwargs)
        return ax

    def _annotate_plot(self, ax, **kwargs):
        ax.set_xlabel(f'PC {self.pcs_to_plot[0] + 1}', labelpad=-15)
        ax.set_ylabel(f'PC {self.pcs_to_plot[1] + 1}', labelpad=-15)
        ax.set_zlabel(f'PC {self.pcs_to_plot[2] + 1}', labelpad=-15)
        ax.set_title(kwargs.get('title', None))
        ax.legend(loc='upper center', ncol=2, frameon=False)
        return ax
