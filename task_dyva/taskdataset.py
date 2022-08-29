"""Classes and utility functions for processing game data.

EbbFlowGameData: Container for data from a single game.
EbbFlowDataset: Subclass of PyTorch Dataset,
    container for data from multiple games.
EbbFlowStats: Subclass of EbbFlowDataset,
    provides extra functionality for analysis.
"""
import os
import random
import copy
import pickle
from itertools import product
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, bernoulli

from . import transforms as T
from .utils import z_pca, exgauss_mle, RemapRTs


class EbbFlowDataset(Dataset):
    """Container for an Ebb and Flow dataset; also provides 
    functionality for processing the data and interfacing with PyTorch.
    Data are stored in two separate formats with a one-to-one correspondence: 
    self.xu contains the model inputs: this is the "continuous" format.
    self.discrete contains the same data in an alternative format
    which facilitates analysis.

    Args
    ----
    experiment_dir (str): Root directory containing info for current
        model training run. 
    params (dict): Processing parameters; see e.g. train_transform_kwargs
        in default config (config/model_config.yaml). 
    preprocessed (dict): Preprocessed game data.
    split (str): Can be either 'train', 'val', or 'test'; 
        split is incorporated into the file name of the processed data 
        upon saving.
    processed_dir (str): Directory in which to save the processed data.
    pre_transform (list of callables, optional): Transformations which 
        are applied before the processed data is saved. 
    transform (list of callables, optional): Transformations which are applied
        'online' on each iteration of the training loop. 
    pre_transform_params (list of dicts, optional): List of parameters to 
        pass to the pre_transforms. This should be a list of dictionaries
        with the same length as pre_transform.
    transform_params (list of dicts, optional): List of parameters to 
        pass to the transforms. This should be a list of dictionaries
        with the same length as transform.
    """

    needs_data_augmentation = ('kde', 'kde_no_switch_cost', 'adaptive_gaussian',
                               'ex_gauss', 'ex_gauss_by_trial_type', 'optimal')

    def __init__(self, experiment_dir, params, preprocessed, split,
                 processed_dir, pre_transform=None, transform=None,
                 pre_transform_params=None, transform_params=None):
        self.experiment_dir = experiment_dir
        self.processed_dir = processed_dir
        self.params = params
        # rename a couple keys
        preprocessed['urt_ms'] = preprocessed.pop('resp_time')
        preprocessed['urespdir'] = preprocessed.pop('resp_dir')
        self.resampling_type = self.params.get('data_augmentation_type', None)
        # Optionally transform data for "optimal" model training
        if self.resampling_type == 'optimal':
            remapper = RemapRTs(preprocessed, params['remap_rt'])
            preprocessed = remapper.remap()
        self.preprocessed = preprocessed
        self.split = split
        # set up data transforms
        self.default_pre = [T.SmoothResponses(), T._Trim()]
        self.supplied_pre = pre_transform
        self.supplied_pre_params = pre_transform_params
        if transform is not None:
            self.transform = T.Compose(
                [t(p) for t, p in zip(transform, transform_params)])

        self.process()  # also saves processed data
        self.xu = torch.load(self.processed_paths[0])  # xu = model inputs
        with open(self.processed_paths[1], 'rb') as path:
            other_data = pickle.load(path)
            self.discrete = other_data['discrete']
            self.game_ids = other_data['game_ids']
            self.resampling_info = other_data['resampling']

    def get_processed_sample(self, idx):
        """Return an EbbFlowGameData instance with data from a single game.

        Args
        ----
        idx (int): Index of the game to return.

        Returns
        -------
        An EbbFlowGameData instance containing the data for this game.

        """

        discrete = {key: vals[idx] for key, vals in self.discrete.items()}
        cnp = self[idx].numpy()
        continuous = {'urespdir': cnp[:, :4], 'point_dir': cnp[:, 4:8],
                      'mv_dir': cnp[:, 8:12], 'task_cue': cnp[:, 12:]}
        game_id = self.game_ids[idx]
        return EbbFlowGameData.processed_format(discrete, continuous, 
                                                self.params, game_id)

    def __getitem__(self, idx):
        # Return a single sample (game) to train the model (continuous format).
        # Called by PyTorch during model training. 
        xu_idx = self.xu[:, idx, :]
        xu_idx = xu_idx if self.transform is None else self.transform(xu_idx)
        return xu_idx

    def __len__(self):
        # Return the number of samples (games) in the dataset.
        return self.xu.shape[1]

    @property
    def processed_paths(self):
        """Return the full paths to the processed data."""
        return [os.path.join(self.processed_dir, f) 
                for f in self.processed_file_names]

    @property
    def processed_file_names(self):
        """Return the names of the processed data files."""
        return [f'{self.split}_model_inputs.pt', 
                f'{self.split}_other_data.pkl']

    def process(self):
        """Prepare an Ebb and Flow dataset for model training. 
        Apply pretransforms and filtering criteria; 
        determine the continuous and discrete formats of the dataset.
        """

        if _files_exist(self.processed_paths):
            return
        os.makedirs(self.processed_dir, exist_ok=True)

        # Do an initial processing run to get info for resampling
        # and/or response smoothing. 
        if ((self.resampling_type in self.needs_data_augmentation)
                or (self.params['smoothing_type'] in 
                    self.needs_data_augmentation)):
            throwaway_data = self._get_preprocessed_games(for_resampling=True)
            [td.standard_prep() for td in throwaway_data]

            # Remove outliers if specified in pre-transform 
            # before resampler estimation
            out_method = self.params['outlier_params'].get('method', None)
            if out_method is not None:
                out_filter = T.FilterOutliers(self.params)
                rs_pre_transform = T.Compose([T._Trim(), out_filter])
            else:
                rs_pre_transform = T.Compose([T._Trim()])

            throwaway_data = [rs_pre_transform(td) for td in throwaway_data]
            throwaway_data = [td for td in throwaway_data if td.is_valid]
            resampling_info, sm_params = self._get_resampling_sm_info(
                throwaway_data)
            self.resampling_info = resampling_info
        else:
            sm_params = copy.deepcopy(self.params)

        # Build pre transform
        self.default_pre[0]._build_sm_kernel(sm_params)
        if self.supplied_pre is not None:
            supplied_pre = [t(p) for t, p in zip(self.supplied_pre, 
                                                 self.supplied_pre_params)]
            self.pre_transform = T.Compose(self.default_pre + supplied_pre)
        else:
            self.pre_transform = T.Compose(self.default_pre)

        # Process each game
        data_list = self._get_preprocessed_games()
        [d.standard_prep() for d in data_list]
        data_list = [self.pre_transform(d) for d in data_list]
        self.excluded_list = [d for d in data_list if not d.is_valid]
        data_list = [d for d in data_list if d.is_valid]
        self._collate(data_list)
        self._save_processed_data()

    def _get_preprocessed_games(self, for_resampling=False):
        data_list = []
        resampling_info = getattr(self, 'resampling_info', None)
        start_times = self.params['start_times']
        if for_resampling:
            upscale_mult = 1
        else:
            upscale_mult = self.params['upscale_mult']

        for start_time, game_ind, _ in product(
                start_times, range(len(self.preprocessed['urt_ms'])),
                range(upscale_mult)):

            preprocessed_game = {key: self.preprocessed[key][game_ind] 
                                 for key in self.preprocessed.keys()}

            data_list.append(
                EbbFlowGameData.preprocessed_format(
                    preprocessed_game, self.params, start_time, 
                    resampling_info=resampling_info))
        return data_list

    def _remove_switch_cost(self, rts_typed):
        # Translate RTs to eliminate switch cost
        con_rt_diff = np.mean(rts_typed[2]) - np.mean(rts_typed[0])
        incon_rt_diff = np.mean(rts_typed[3]) - np.mean(rts_typed[1])
        orig_mean_typed_rt = np.mean([np.mean(rts_typed[i]) for i in range(4)])
        rts_typed[2] = np.array(rts_typed[2]) - con_rt_diff
        rts_typed[3] = np.array(rts_typed[3]) - incon_rt_diff
        # Translate RTs again so that mean RT is the same
        new_mean_typed_rt = np.mean([np.mean(rts_typed[i]) for i in range(4)])
        mean_rt_diff = orig_mean_typed_rt - new_mean_typed_rt
        for ttype in range(4):
            rts_typed[ttype] += mean_rt_diff
        return rts_typed

    def _get_resampling_sm_info(self, data):
        # Trial types: 
        # 0 = congruent + stay
        # 1 = incongruent + stay
        # 2 = congruent + switch
        # 3 = incongruent + switch
        resampling_dists = {}
        acc = {}
        rts_typed = {}
        sm_params = {'step_size': self.params['step_size'],
                     'smoothing_type': self.params.get('smoothing_type', 
                                                       'gaussian'),
                     'kernel_sd': self.params.get('kernel_sd', 50),
                     'params': {}}

        if self.params['smoothing_type'] == 'ex_gauss':
            # Smooth trials with an exGauss kernel estimated from all RTs
            rts = []
            for d in data:
                rts.extend(d.discrete['urt_ms'])
            sm_params['ex_gauss_rv'] = exgauss_mle(rts)
        elif self.params['smoothing_type'] == 'optimal':
            sm_params['remap_rt'] = self.params['remap_rt']
            sm_params['optimal_min_rt'] = self.params['optimal_min_rt']
            sm_params['post_resp_buffer'] = self.params['post_resp_buffer']
        elif self.params['smoothing_type'] == 'optimal_short':
            sm_params['remap_rt'] = self.params['remap_rt']
            sm_params['kernel_width'] = self.params['optimal_kernel_width']

        for ttype in range(4):
            this_rts = []
            this_correct = []
            for d in data:
                d.get_extra_stats()
                this_rts.extend(d._get_field_by_trial_type(ttype, 'urt_ms'))
                this_correct.extend(d._get_field_by_trial_type(
                    ttype, 'ucorrect'))
            rts_typed[ttype] = this_rts

        # Resampling info
        bw = self.params.get('data_aug_kernel_bandwidth', 0.25)
        if self.resampling_type == 'kde_no_switch_cost':
            rts_typed = self._remove_switch_cost(rts_typed)

        for ttype in range(4):
            this_rts = rts_typed[ttype]
            if self.resampling_type in ['kde', 'kde_no_switch_cost']:
                this_resampling = gaussian_kde(this_rts, bw_method=bw)
            else:
                this_resampling = None

            # Smoothing info
            if self.params['smoothing_type'] == 'adaptive_gaussian':
                this_sm = np.std(this_rts)
            elif self.params['smoothing_type'] == 'ex_gauss_by_trial_type':
                this_sm = exgauss_mle(this_rts)
            elif self.params['smoothing_type'] == 'kde':
                bw = self.params.get('data_aug_kernel_bandwidth', 0.25)
                this_sm = gaussian_kde(this_rts, bw_method=bw)
            else:
                this_sm = None

            resampling_dists[ttype] = this_resampling
            sm_params['params'][ttype] = this_sm
            acc[ttype] = np.mean(this_correct)
            if resampling_dists[0] is not None:
                resampling_info = {'rts': resampling_dists, 'acc': acc}
            else:
                resampling_info = None

        return resampling_info, sm_params

    def _save_processed_data(self):
        other_data = {'discrete': self.discrete,
                      'excluded': self.excluded_list,
                      'resampling': getattr(self, 'resampling_info', None),
                      'game_ids': self.game_ids}
        torch.save(self.xu, self.processed_paths[0])
        with open(self.processed_paths[1], 'xb') as path:
            dill.dump(other_data, path, protocol=4)

    def _collate(self, data_list):
        # Continuous format (model inputs)
        con_keys = ['urespdir', 'point_dir', 'mv_dir', 'task_cue']
        xu_split = [torch.cat([torch.tensor(d.continuous[key]).unsqueeze(1) 
                               for d in data_list], 1)
                    for key in con_keys]
        self.xu = torch.cat([d for d in xu_split], 2).to(dtype=torch.float32)

        # Discrete format
        disc_keys = data_list[0].discrete_fields
        self.discrete = {key: [d.discrete[key] for d in data_list]
                         for key in disc_keys}
        self.game_ids = [d.game_id for d in data_list]


class EbbFlowStats(EbbFlowDataset):
    """Extends EbbFlowDataset with extra functionality for 
    analyzing user and model behavior.

    Args
    ----
    output_rates (PyTorch tensor): The models outputs (responses for each of 
        the four directions). Has dimensions n_timesteps x n_samples x 4.
    dataset (EbbFlowDataset instance): The dataset to be analyzed.
    latents (PyTorch tensor, optional): The model latent state variables.
        Has dimensions n_timesteps x n_samples x latent_dim.
    n_pcs (int, optional): The number of principal components to keep
        in the PCA transformed latent state variables (self.pca_latents).
    **kwargs (optional): Extra options to be supplied for calls to
        EbbFlowGameData.get_extra_stats().
    """

    def __init__(self, output_rates, dataset, latents=None,
                 n_pcs=3, **kwargs):
        self.rates = output_rates.cpu().detach().numpy()
        self.xu = dataset.xu
        self.discrete = dataset.discrete
        self.transform = dataset.transform
        self.params = dataset.params
        self.step = self.params['step_size']
        self.game_ids = dataset.game_ids
        td_kwargs = {'t_pre': 100, 't_post': 1600}
        td_kwargs.update(kwargs)
        self.trial_data_kwargs = td_kwargs
        self.n_pre = np.round(td_kwargs['t_pre'] / self.step).astype('int')
        self.n_post = np.round(td_kwargs['t_post'] / self.step).astype('int')
        self.t_axis = self.step * np.arange(-self.n_pre, self.n_post, 1) 
        if latents is not None:
            self.latents = latents.cpu().detach().numpy()
            pca_latents, explained_var, pca_obj = z_pca(self.latents, n_pcs)
            self.pca_latents = pca_latents
            self.pca_explained_var = explained_var
            self.pca_obj = pca_obj
        else:
            self.latents = None
            self.pca_latents = None
            self.pca_explained_var = None
            self.pca_obj = None
        self.windowed = None
        self._get_trial_data()

    def _get_trial_data(self):
        # Transform the discrete dataset to a pandas data frame;
        # get extra stats in the process. Also window and concatenate
        # the output rates and optionally the model latents from each trial.
        dfs = []
        for d in range(self.rates.shape[1]):
            this_game = self.get_processed_sample(d)
            this_rates = np.squeeze(self.rates[:, d, :])
            if self.latents is not None:
                this_latents = np.squeeze(self.latents[:, d, :])
                this_pca_latents = np.squeeze(self.pca_latents[:, d, :])
                win_vars = this_game.get_extra_stats(
                    output_rates=this_rates, latents=this_latents, 
                    pca_latents=this_pca_latents, **self.trial_data_kwargs)
            else:
                win_vars = this_game.get_extra_stats(
                    output_rates=this_rates, **self.trial_data_kwargs)
            self._concat_windowed(win_vars)
            dfs.append(this_game._to_pandas())
        self.df = pd.concat(dfs, ignore_index=True)

    def _concat_windowed(self, win_vars):
        if self.windowed is None:
            self.windowed = win_vars
        else:
            for key, val in win_vars.items():
                if val is None:
                    continue
                self.windowed[key] = np.concatenate(
                    (self.windowed[key], val), 1)

    def select(self, df=None, **kwargs):
        """Select a subset of trials using the criteria specified in **kwargs.

        Args
        ----
        df (pandas DataFrame, optional): Data to be selected from. 
            If None (default), self.df is used. 
        **kwargs (optional): Selection criteria specified as a dictionary.
            Each key should correspond to one of the fields in the discrete
            data format. 

        Returns
        -------
        trial_inds (NumPy array): The indices of the selected trials. 

        Example
        -------
        Select all congruent switch trials:
            >>> trial_inds = self.select(**{'is_switch': 1, 'is_congruent': 1})
        """

        select_df = self.df if df is None else df
        query_str = ''
        for key, val in kwargs.items():
            query_str += f'({key} == {val}) & '
        query_str = query_str[:-3]
        trial_selection = select_df.query(query_str)
        trial_inds = trial_selection.index.to_numpy()
        return trial_inds

    def switch_cost(self):
        """Calculate switch cost summary statistics for user and model.

        Returns
        -------
        stats (dict): Switch cost statistics; has the following keys:
            u_switch_cost: The user's mean response time on switch trials
                minus the user's mean response time on stay trials (ms).
            m_switch_cost: As above, but for the model's responses.
            u_acc_switch_cost: The user's mean accuracy on stay trials
                minus the user's mean accuracy on switch trials.
            m_acc_switch_cost: As above, but for the model's responses.
        """

        stats = {}
        u_stay_inds = self.select(**{'is_switch': 0, 'ucorrect': 1, 
                                     'u_prev_correct': 1})
        m_stay_inds = self.select(**{'is_switch': 0, 'mcorrect': 1, 
                                     'm_prev_correct': 1})
        u_switch_inds = self.select(**{'is_switch': 1, 'ucorrect': 1, 
                                       'u_prev_correct': 1})
        m_switch_inds = self.select(**{'is_switch': 1, 'mcorrect': 1, 
                                       'm_prev_correct': 1})
        # response times
        u_stay_rts = self.df['urt_ms'][u_stay_inds]
        m_stay_rts = self.df['mrt_ms'][m_stay_inds]
        u_switch_rts = self.df['urt_ms'][u_switch_inds]
        m_switch_rts = self.df['mrt_ms'][m_switch_inds]
        stats['u_switch_cost'] = u_switch_rts.mean() - u_stay_rts.mean()
        stats['m_switch_cost'] = m_switch_rts.mean() - m_stay_rts.mean()
        # accuracy
        u_acc_stay_inds = self.select(**{'is_switch': 0, 'u_prev_correct': 1})
        m_acc_stay_inds = self.select(**{'is_switch': 0, 'm_prev_correct': 1})
        u_acc_switch_inds = self.select(**{'is_switch': 1, 'u_prev_correct': 1})
        m_acc_switch_inds = self.select(**{'is_switch': 1, 'm_prev_correct': 1})
        u_stay_c = self.df['ucorrect'][u_acc_stay_inds]
        m_stay_c = self.df['mcorrect'][m_acc_stay_inds]
        u_switch_c = self.df['ucorrect'][u_acc_switch_inds]
        m_switch_c = self.df['mcorrect'][m_acc_switch_inds]
        stats['u_acc_switch_cost'] = u_stay_c.mean() - u_switch_c.mean()
        stats['m_acc_switch_cost'] = m_stay_c.mean() - m_switch_c.mean()

        # Calculate slightly differently for early stopping metrics: use all trials
        stay_inds_estop = self.select(**{'is_switch': 0})
        switch_inds_estop = self.select(**{'is_switch': 1})
        u_stay_rts_estop = self.df['urt_ms'][stay_inds_estop]
        m_stay_rts_estop = self.df['mrt_ms'][stay_inds_estop]
        u_switch_rts_estop = self.df['urt_ms'][switch_inds_estop]
        m_switch_rts_estop = self.df['mrt_ms'][switch_inds_estop]
        stats['u_switch_cost_estop'] = u_switch_rts_estop.mean() - u_stay_rts_estop.mean()
        stats['m_switch_cost_estop'] = m_switch_rts_estop.mean() - m_stay_rts_estop.mean()
        return stats

    def congruency_effect(self):
        """Calculate congruency effect summary statistics for user and model.

        Returns
        -------
        stats (dict): Congruency effect statistics; has the following keys:
            u_con_effect: The user's mean response time on incongruent trials
                minus the user's mean response time on congruent trials.
            m_con_effect: As above, but for the model's responses. 
            u_acc_con_effect: The user's mean accuracy on congruent trials
                minus the user's mean accuracy on incongruent trials.
            m_acc_con_effect: As above, but for the model's responses. 
        """

        stats = {}
        u_con_inds = self.select(**{'is_congruent': 1, 'ucorrect': 1})
        m_con_inds = self.select(**{'is_congruent': 1, 'mcorrect': 1})
        u_incon_inds = self.select(**{'is_congruent': 0, 'ucorrect': 1})
        m_incon_inds = self.select(**{'is_congruent': 0, 'mcorrect': 1})
        # response times
        u_con_rts = self.df['urt_ms'][u_con_inds]
        m_con_rts = self.df['mrt_ms'][m_con_inds]
        u_incon_rts = self.df['urt_ms'][u_incon_inds]
        m_incon_rts = self.df['mrt_ms'][m_incon_inds]
        stats['u_con_effect'] = u_incon_rts.mean() - u_con_rts.mean()
        stats['m_con_effect'] = m_incon_rts.mean() - m_con_rts.mean()
        # accuracy
        acc_con_inds = self.select(**{'is_congruent': 1})
        acc_incon_inds = self.select(**{'is_congruent': 0})
        u_con_c = self.df['ucorrect'][acc_con_inds]
        m_con_c = self.df['mcorrect'][acc_con_inds]
        u_incon_c = self.df['ucorrect'][acc_incon_inds]
        m_incon_c = self.df['mcorrect'][acc_incon_inds]
        stats['u_acc_con_effect'] = u_con_c.mean() - u_incon_c.mean()
        stats['m_acc_con_effect'] = m_con_c.mean() - m_incon_c.mean()

        # Calculate slightly differently for early stopping metrics: use all trials
        con_inds_estop = self.select(**{'is_congruent': 1})
        incon_inds_estop = self.select(**{'is_congruent': 0})
        u_con_rts_estop = self.df['urt_ms'][con_inds_estop]
        m_con_rts_estop = self.df['mrt_ms'][con_inds_estop]
        u_incon_rts_estop = self.df['urt_ms'][incon_inds_estop]
        m_incon_rts_estop = self.df['mrt_ms'][incon_inds_estop]
        stats['u_con_effect_estop'] = u_incon_rts_estop.mean() - u_con_rts_estop.mean()
        stats['m_con_effect_estop'] = m_incon_rts_estop.mean() - m_con_rts_estop.mean()
        return stats

    def get_stats(self):
        """Calculate summary statistics for user and model behavior.

        Returns
        -------
        stats (dict): Summary statistics on the accuracy and
            response times for both user and model (e.g. the switch cost 
            and congruency effect).
        """

        stats = {}
        stats.update(self.switch_cost())
        stats.update(self.congruency_effect())
        stats['u_accuracy'] = self.df['ucorrect'].mean()
        stats['m_accuracy'] = self.df['mcorrect'].mean()
        urts = self.df['urt_ms'][self.select(**{'ucorrect': 1})]
        mrts = self.df['mrt_ms'][self.select(**{'mcorrect': 1})]
        stats['u_mean_rt'] = urts.mean()
        stats['m_mean_rt'] = mrts.mean()
        stats['u_rt_sd'] = urts.std()
        stats['m_rt_sd'] = mrts.std()
        self.summary_stats = stats
        return stats


class EbbFlowGameData():
    """Container for data from a single Ebb and Flow game. Also has support 
    for processing game data. This can be instantiated using one of two class 
    constructors: 
    preprocessed_format: For data that needs to be processed
        (transformed, filtered, etc.). 
    processed_format: For data that has already been processed. 

    Data are maintained in two formats which have a one-to-one 
    correspondence: a discrete format, which is easier to analyze, and a 
    continuous format, which is the format supplied to the model. 
    If instantiated with the preprocessed constructor, discrete and continuous 
    are initialized as empty arrays and populated sequentially during 
    processing. 

    Args
    ----
    preprocessed (dict): Preprocessed game data. Set to None
        if instantiated with the processed_format constructor.
    discrete (dict): The discrete format of the data. See discrete_fields. 
    continuous (dict): The continuous format of the data. 
        See continuous_fields.
    params (dict): Processing parameters; see e.g. train_transform_kwargs
        in default config (config/model_config.yaml). 
    start_time (int): The time within the game to start collecting trials.
        Set to None if instantiated with the processed_format constructor.
    resampling_info (dict): Information used to generate resampled responses.
        Set to None if instantiated with the processed_format constructor.
    game_id (int): The ID of this gameplay as it is stored 
        in the Lumosity database. 
    """

    continuous_fields = ('urespdir', 'point_dir', 'mv_dir', 'task_cue')
    discrete_fields = ('onset', 'offset', 'urespdir', 'point_dir',
                       'mv_dir', 'task_cue', 'urt_samples', 'urt_ms', 
                       'trial_type')
    stats_fields = ('prev_point_dir', 'prev_mv_dir', 'prev_task_cue',
                    'm_prev_correct', 'u_prev_correct',
                    'is_switch', 'is_congruent', 'correct_dir',
                    'mrespdir', 'mcorrect', 'ucorrect', 'mrt_ms', 'mrt_abs')
    dims = (4, 4, 4, 2)
    direction_labels = ('L', 'R', 'U', 'D')
    task_labels = ('M', 'P')
    extra_time_for_smooth = 2500  # ms
    supported_resampling = ('kde')

    def __init__(self, preprocessed, discrete, continuous, 
                 params, start_time, resampling_info, 
                 game_id):
        self.preprocessed = preprocessed
        self.discrete = discrete
        self.continuous = continuous
        self.start_time = start_time
        self.step = params['step_size']
        self.num_steps_short_win = int(
            np.rint(params['duration'] / self.step))
        self.num_steps_long_win = int(
            np.rint((params['duration'] 
                     + self.extra_time_for_smooth) / self.step))
        self.max_t = params['duration'] + self.extra_time_for_smooth
        self.params = params
        self.resampling_type = params.get('data_augmentation_type', None)
        self.resampling_info = resampling_info
        # If trials are being resampled during processing, rs_frac is 
        # the proportion of games which are resampled. The rest are 
        # not resampled (i.e., the original sequence of trials is preserved). 
        rs_frac = params.get('aug_resample_frac', 0.75)
        self.do_resampling = bernoulli(rs_frac).rvs(1)[0]
        self.is_valid = True
        self.game_id = game_id
        # Optionally match the accuracy of the resampled data 
        # to the user's accuracy.
        self.match_accuracy = params.get('match_accuracy', False)
        # rt_tol is the min time after stim onset that a response can occur
        # (see _get_model_rt).
        self.rt_tol = 100 / self.step  # samples

    @property
    def _n_trials(self):
        if self.resampling_info is None or not self.do_resampling:
            return len(self.preprocessed['urespdir'])
        else:
            return float('inf')

    @classmethod
    def processed_format(cls, discrete, continuous, params, game_id):
        """Return a class instance for game data that has already 
        been processed. See class docstring.
        """

        preprocessed = None
        start_time = None
        resampler = None
        return cls(preprocessed, discrete, continuous, params, start_time,
                   resampler, game_id)

    @classmethod
    def preprocessed_format(cls, preprocessed, params, start_time, 
                            resampling_info=None):
        """Return a class instance for game data that needs to be processed.
        See class docstring.
        """

        discrete, continuous = cls._initialize_arrays(params)
        game_id = preprocessed['game_result_id']
        return cls(preprocessed, discrete, continuous, params, start_time, 
                   resampling_info, game_id)

    @classmethod
    def _initialize_arrays(cls, params):
        discrete = defaultdict(list)
        num_samples = int(np.rint((params['duration'] 
                                   + cls.extra_time_for_smooth)
                                  / params['step_size']))
        continuous = {key: np.zeros((num_samples, d))
                      for key, d in zip(cls.continuous_fields[1:], 
                                        cls.dims[1:])}
        continuous['urespdir'] = np.zeros((num_samples, 4, 4))
        return discrete, continuous

    def standard_prep(self):
        """Process the game data; optionally resample trials. 
        This is only called if instantiated with the preprocessed_format 
        constructor. The discrete and continuous arrays are populated 
        sequentially one trial at a time. 
        """

        trial_ind = self._check_first_trial()
        if np.isnan(trial_ind):
            self.is_valid = False
            return

        abs_offset_ms = 0
        if self.resampling_info is None:
            if trial_ind > 0:
                prev_cue_str = self.preprocessed['task_cue'][trial_ind - 1]
                prev_cue = self._map_str_to_num(prev_cue_str, 'task_cue')
            else: 
                self.is_valid = False
                return
        else:
            prev_cue = self._map_str_to_num(
                random.sample(self.task_labels, 1)[0],
                'task_cue')
        while trial_ind < self._n_trials:
            trial_info, abs_offset_ms = self._get_trial_info_preprocessed(
                trial_ind, abs_offset_ms, prev_cue)
            prev_cue = trial_info['task_cue']
            trial_ind += 1

            if trial_info['offset'] < self.num_steps_short_win: 
                self._update_discrete(trial_info)
            if trial_info['offset'] < self.num_steps_long_win: 
                self._update_continuous(trial_info)
            if abs_offset_ms > self.max_t:
                break

        if len(self.discrete['mv_dir']) < self.params['min_trials']:
            self.is_valid = False

    def _map_str_to_num(self, str_val, key):
        # Map stimulus/response direction string to numeric value
        if key in ['urespdir', 'mv_dir', 'point_dir']:
            labels = self.direction_labels
        else:
            labels = self.task_labels
        num_val = [ind for ind, val in enumerate(labels) if val == str_val]
        return num_val[0]

    def _get_trial_info_preprocessed(self, trial_ind, abs_offset_ms, 
                                     prev_cue):
        if self.resampling_info is not None and self.do_resampling:
            trial_info = {key: None for key in self.continuous_fields}
            trial_info, prev_cue = self._resample_trial(trial_info, prev_cue)
            trial_info = self._get_resampler_trial_type(trial_info, prev_cue)
            if self.match_accuracy:
                trial_info = self._adjust_trial_response(trial_info)
        else:
            trial_info = {key: self._map_str_to_num(
                self.preprocessed[key][trial_ind], key)
                for key in self.continuous_fields}
            trial_info['urt_ms'] = self.preprocessed['urt_ms'][trial_ind]
            trial_info = self._get_resampler_trial_type(trial_info, prev_cue)

        # Floor is used to ensure there is a gap between consecutive stimuli
        trial_info['onset'] = int(np.floor(abs_offset_ms / self.step))
        trial_info['offset'] = int(
            np.floor((abs_offset_ms
                      + trial_info['urt_ms']
                      + self.params['post_resp_buffer']) 
                     / self.step)) - 1
        trial_info['urt_samples'] = int(
            np.rint(trial_info['urt_ms'] / self.step))
        abs_offset_ms += trial_info['urt_ms'] + self.params['post_resp_buffer']
        return trial_info, abs_offset_ms

    def _resample_trial(self, trial_info, prev_cue):
        # Trial types: 
        # 0 = congruent + stay
        # 1 = incongruent + stay
        # 2 = congruent + switch
        # 3 = incongruent + switch

        if np.isnan(prev_cue):
            prev_cue = random.sample([0, 1], 1)[0]
        # Randomly sample condition
        con_ind = np.random.choice(4)
        if self.resampling_type in ['kde', 'kde_no_switch_cost']:
            this_dist = self.resampling_info['rts'][con_ind]
            new_rt_ms = this_dist.resample(size=1)
        elif self.resampling_type == 'optimal':
            new_rt_ms = self.params['remap_rt']

        # Stay vs. switch
        if con_ind in [2, 3]:
            # Switch trial
            if prev_cue == 0:
                new_cue = 1
            else:
                new_cue = 0
        else:
            # Stay trial
            new_cue = prev_cue

        # Congruent vs. incongruent
        new_mv_dir = self._map_str_to_num(
            random.sample(self.direction_labels, 1)[0],
            'mv_dir')
        if con_ind in [0, 2]:
            # Congruent trial
            new_pt_dir = new_mv_dir
        else:
            # Incongruent trial
            other_dirs = [i for i in range(4) if i != new_mv_dir]
            new_pt_dir = random.sample(other_dirs, 1)[0]

        # Response: set to correct dir; optionally adjusted later
        new_uresp = self._get_correct_dir(new_cue, new_mv_dir, new_pt_dir)

        new_data = {'urt_ms': new_rt_ms[0][0], 'task_cue': new_cue, 
                    'mv_dir': new_mv_dir, 'point_dir': new_pt_dir,
                    'urespdir': new_uresp}
        trial_info.update(new_data)
        return trial_info, prev_cue

    def _adjust_trial_response(self, trial_info):
        # Randomly change the resampled response to an incorrect direction
        # at a rate determined by the user's accuracy for this trial type.
        con_acc = self.resampling_info['acc'][trial_info['trial_type']]
        is_correct = bernoulli.rvs(con_acc)
        new_correct_dir = self._get_correct_dir(
            trial_info['task_cue'], trial_info['mv_dir'], 
            trial_info['point_dir'])
        if is_correct:
            trial_info['urespdir'] = new_correct_dir
        else:
            incorrect_dirs = [i for i in range(4) if i != new_correct_dir]
            trial_info['urespdir'] = random.sample(incorrect_dirs, 1)
        return trial_info

    def _get_resampler_trial_type(self, trial_info, prev_cue):
        is_congruent = self._is_congruent(trial_info['mv_dir'], 
                                          trial_info['point_dir'])
        if np.isnan(prev_cue):
            # No previous trial to calculate switch;
            # treat as a switch for calculating resampling dists.
            is_switch = np.nan
        else:
            is_switch = self._is_switch(trial_info['task_cue'], prev_cue)

        if np.isnan(is_switch):
            trial_info['trial_type'] = np.nan
            self.is_valid = False
        elif is_congruent and not is_switch:
            trial_info['trial_type'] = 0
        elif is_congruent and is_switch:
            trial_info['trial_type'] = 2
        elif not is_congruent and not is_switch:
            trial_info['trial_type'] = 1
        elif not is_congruent and is_switch:
            trial_info['trial_type'] = 3
        return trial_info

    def _check_first_trial(self):
        game_t_offs = np.array(self.preprocessed['time_offset'])
        try:
            trial_ind = np.nonzero(game_t_offs >= self.start_time)[0][0]
        except IndexError:
            trial_ind = np.nan
        return trial_ind

    def _update_discrete(self, trial_info):
        for key in self.discrete_fields:
            self.discrete[key].append(trial_info[key])

    def _update_continuous(self, trial_info):
        for key in ['mv_dir', 'point_dir', 'task_cue']:
            self.continuous[key][trial_info['onset']:trial_info['offset'],
                                 trial_info[key]] = 1

        abs_rt = trial_info['onset'] + trial_info['urt_samples']
        self.continuous['urespdir'][abs_rt, trial_info['urespdir'], 
                                    trial_info['trial_type']] = 1

    def get_extra_stats(self, output_rates=None, latents=None, 
                        pca_latents=None, **kwargs):
        """Add extra information to the discrete format 
        (see self.stats_fields): congruency, stay/switch, model output RTs, 
        switch cost,congruency effect, and previous trial info. 
        Also window the output rates, model latents, and PCA-tranformed
        model latents for each trial. 

        Args
        ----
        output_rates (NumPy array, optional): The rates for each response
            direction generated in a forward pass of the model. 
        latents (NumPy array, optional): The latent state variables generated
            in a forward pass of the model. 
        pca_latents (NumPy array, optional): PCA-transformed latent
            state variables. 
        kwargs (dict, optional): Optional parameters which determine the 
            length of each trial in the windowed variables:
            t_pre (int): Time in ms prior to stimulus onset (default 100).
            t_post (int): Time in ms after stimulus onset (default 1600). 

        Returns
        -------
        win_vars (dict): Windowed rates, model latents, and PCA-transformed
            model latents for each trial. 
        """

        win_rates, win_latents, win_pca_latents = None, None, None
        if output_rates is not None or latents is not None:
            t_pre = kwargs.get('t_pre', 100)  # ms
            t_post = kwargs.get('t_post', 1600)  # ms
            n_pre = np.round(t_pre / self.step).astype('int')
            n_post = np.round(t_post / self.step).astype('int')
            win_length = n_pre + n_post
            win_rates = np.zeros((win_length, 0, 4))
        if latents is not None:
            win_latents = np.zeros((win_length, 0, latents.shape[1]))
            win_pca_latents = np.zeros((win_length, 0, pca_latents.shape[1]))

        n_trials = len(self.discrete['mv_dir'])
        self.discrete.update({key: [] for key in self.stats_fields})
        for n in range(n_trials):
            tr = {key: np.nan for key in self.stats_fields}
            if n != 0:
                tr.update(prev_trial_info)

            tr.update({key: self.discrete[key][n] 
                       for key in self.discrete_fields})

            # Stimulus info
            tr['correct_dir'] = self._get_correct_dir(tr['task_cue'], 
                                                      tr['mv_dir'], 
                                                      tr['point_dir'])
            tr['is_congruent'] = self._is_congruent(tr['mv_dir'], 
                                                    tr['point_dir'])
            if n != 0:
                tr['is_switch'] = self._is_switch(tr['task_cue'], 
                                                  tr['prev_task_cue'])

            # User response info
            tr['ucorrect'] = self._is_correct(tr['urespdir'], 
                                              tr['correct_dir'])

            # Model response info
            if output_rates is not None:
                tr['mrt_ms'], tr['mrt_abs'], tr['mrespdir'] = \
                    self._get_model_rt(tr['onset'], tr['offset'], 
                                       output_rates)
                tr['mcorrect'] = self._is_correct(tr['mrespdir'], 
                                                  tr['correct_dir'])

            # Window rates and model latents for current trial
            if output_rates is not None:
                this_win_rates = self._get_windowed(
                    tr['onset'], output_rates, win_length, n_pre, n_post)
                win_rates = np.concatenate((win_rates, this_win_rates), 1)
            if latents is not None:
                this_win_latents = self._get_windowed(
                    tr['onset'], latents, win_length, n_pre, n_post)
                win_latents = np.concatenate(
                    (win_latents, this_win_latents), 1)
                this_win_pca_latents = self._get_windowed(
                    tr['onset'], pca_latents, win_length, n_pre, n_post)
                win_pca_latents = np.concatenate(
                    (win_pca_latents, this_win_pca_latents), 1)

            # Update discrete format and previous trial info
            self._add_stats_to_discrete(tr)
            prev_trial_info = {'prev_task_cue': tr['task_cue'], 
                               'prev_point_dir': tr['point_dir'],
                               'prev_mv_dir': tr['mv_dir'],
                               'm_prev_correct': tr['mcorrect'],
                               'u_prev_correct': tr['ucorrect']}

        win_vars = {'rates': win_rates,
                    'latents': win_latents,
                    'pca_latents': win_pca_latents}

        return win_vars

    def _get_windowed(self, onset, data, win_length, n_pre, n_post):
        dim = data.shape[1]
        windowed = data[onset - n_pre:onset + n_post, :]
        if windowed.shape[0] < win_length:  # zero pad
            windowed = np.append(windowed, np.zeros(
                (win_length - windowed.shape[0], dim)), 0)
        return np.expand_dims(windowed, 1)

    def _to_pandas(self):
        dfs = []
        n_trials = len(self.discrete['mv_dir'])
        for n in range(n_trials):
            trial = {key: [val[n]] for key, val in self.discrete.items()}
            dfs.append(pd.DataFrame(trial))
        return pd.concat(dfs, ignore_index=True)

    def _add_stats_to_discrete(self, data):
        for key in self.stats_fields:
            self.discrete[key].append(data[key])

    def plot(self, rates=None, do_plot=False, stim_ylims=None, 
             resp_ylims=None):
        """Plot the continuous representation of the stimuli and responses
        for this game. For the responses, the continuous format 
        of the user's responses is plotted (used to train the model). 
        The responses generated by the model can optionally also be plotted. 

        Args
        ----
        rates (NumPy array, optional): The responses generated by the model.
        do_plot (Boolean, optional): If True, the figure is plotted. 

        Returns
        -------
        fig (matplotlib Figure): The generated figure. 
        axes (matplotlib AxesSubplot): The figure axes, can be used for 
            further tweaking. 
        """

        textsize = 14
        figsize = (10, 18)
        colors = ['royalblue', 'crimson', 'forestgreen', 'orange']
        n_time = self.continuous['point_dir'].shape[0]
        x_plot = np.arange(n_time) * self.step
        stimulus_ylims = [-0.5, 1.5] if stim_ylims is None else stim_ylims
        response_ylims = [-0.2, 1.2] if resp_ylims is None else resp_ylims

        fig, axes = plt.subplots(14, 1, figsize=figsize)
        # Pointing stimuli
        for d in range(4):
            sns.lineplot(x=x_plot, y=self.continuous['point_dir'][:, d], 
                         ax=axes[d], color=colors[0])
            axes[d].set_ylabel(self.direction_labels[d], fontsize=textsize)
            if d == 0:
                axes[d].set_title('pointing stimuli', fontsize=textsize)

        # Moving stimuli
        for d in range(4):
            sns.lineplot(x=x_plot, y=self.continuous['mv_dir'][:, d], 
                         ax=axes[d + 4], color=colors[1])
            axes[d + 4].set_ylabel(self.direction_labels[d], fontsize=textsize)
            if d == 0:
                axes[d + 4].set_title('moving stimuli', fontsize=textsize)

        # Task cues
        for d in range(2):
            sns.lineplot(x=x_plot, y=self.continuous['task_cue'][:, d], 
                         ax=axes[d + 8], color=colors[2])
            axes[d + 8].set_ylabel(self.task_labels[d], fontsize=textsize)
            if d == 0:
                axes[d + 8].set_title('task cues', fontsize=textsize)

        # Responses
        for d in range(4):
            # User
            user_resp = self.continuous['urespdir'][:, d]
            sns.lineplot(x=x_plot, y=user_resp, ax=axes[d + 10], color='k', 
                         zorder=1, label='user')

            if rates is not None:
                # Model
                sns.lineplot(x=x_plot, y=rates[:, d], ax=axes[d + 10], 
                             color=colors[3], zorder=2, label='model')
                this_mrts = [rt for rt, rdir in zip(self.discrete['mrt_abs'],
                                                    self.discrete['mrespdir'])
                             if rdir == d]
                #this_mrts = [20*(rt+on) for rt, on, rdir in zip(self.discrete['urt_samples'],
                #                                    self.discrete['onset'],
                #                                    self.discrete['urespdir'])
                #             if rdir == d]

                sns.scatterplot(x=x_plot[this_mrts], 
                                y=rates[this_mrts, d], s=15, 
                                ax=axes[d + 10], zorder=3, 
                                label='response times')
                axes[d + 10].set_ylabel(self.direction_labels[d], 
                                        fontsize=textsize)

            if d == 0:
                axes[d + 10].set_title('responses', fontsize=textsize)
                axes[d + 10].legend(loc=1, fontsize=textsize - 5, 
                                    frameon=False,
                                    columnspacing=1.25, ncol=3)
            else:
                axes[d + 10].get_legend().remove()

        # Adjust
        [axes[d].set_ylim(stimulus_ylims) for d in range(10)]
        [axes[d].set_ylim(response_ylims) for d in range(10, 14)]
        [axes[d].set_yticks([]) for d in range(14)]
        [axes[d].set_xticklabels([]) for d in range(13)]
        t_max = n_time * self.step
        axes[13].set_xticks(np.arange(0, t_max + 1000, 1000))
        axes[13].tick_params(axis="x", labelsize=textsize)
        axes[13].set_xlabel('time (ms)', fontsize=textsize)
        [axes[d].set_xlim([0, x_plot[-1]]) for d in range(14)]

        plt.tight_layout()
        if do_plot:
            plt.show()
        return fig, axes

    def _get_correct_dir(self, cue, mv, pt):
        return mv if cue == 0 else pt

    def _is_congruent(self, mv, pt):
        return 1 if mv == pt else 0

    def _is_switch(self, cue, prev_cue):
        return 0 if cue == prev_cue else 1

    def _is_correct(self, respdir, correct_dir):
        return 1 if respdir == correct_dir else 0

    def _get_model_rt(self, onset, offset, rates):
        # Get the response time for the current trial from the generated rates.
        win_on = np.round(onset + self.rt_tol).astype('int')
        win_rates = rates[win_on:(offset + 1), :]
        max_ind = np.unravel_index(np.argmax(
            win_rates, axis=None), win_rates.shape)
        mrespdir = max_ind[1]
        if self.params['rt_method'] == 'max':
            # Response time is calculated as the time at which the maximum 
            # activation occurs across all four response directions, 
            # within the stimulus window. 
            mrt_abs = max_ind[0] + win_on  # samples, relative to 0
        elif self.params['rt_method'] == 'center_of_mass':
            # Response time is calculated as the center of mass
            # of the rate with the highest activation. 
            max_rate = win_rates[:, mrespdir]
            csum = np.cumsum(max_rate)
            com = np.nonzero(csum >= (csum[-1] / 2))[0][0]
            mrt_abs = win_on + com  # samples, relative to zero
        mrt = self.step * (mrt_abs - onset)  # ms
        return mrt, mrt_abs, mrespdir

    def _get_field_by_trial_type(self, trial_type, key):
        inds = [i for i, val in enumerate(self.discrete['trial_type'])
                if val == trial_type]
        return [self.discrete[key][i] for i in inds]


def _files_exist(files):
    # Return False if len(files) is zero or if any of the files do not exist.
    return len(files) != 0 and all([os.path.exists(f) for f in files])
