import shutil
import os
import pickle
import random

import numpy as np
from scipy.signal import find_peaks

from .experiment import filter_nth_play, Experiment


class SetUpTests():
    # Streamline setting up experiments for tests
    rand_seed = 11345

    def __init__(self, test_dir, raw_data_dir, raw_data_fn, **expt_kwargs):
        self.test_dir = test_dir
        self.raw_data_dir = raw_data_dir
        self.raw_data_fn = raw_data_fn
        self.expt_kwargs = expt_kwargs
        random.seed(self.rand_seed)
        np.random.seed(self.rand_seed)

    def get_pre_datasets(self, expt):
        # Get the original preprocessed splits from an experiment
        with open(expt.data_path, 'rb') as handle:
            pre_processed = pickle.load(handle)
        nth_range = expt.config_params['data_params']['nth_play_range']
        filtered_data = filter_nth_play(pre_processed, nth_range)
        pre_datasets = expt._split_train_val_test(filtered_data)
        return pre_datasets

    def make_experiment(self):
        processed_dir = os.path.join(self.test_dir, 'processed')
        expt = Experiment(self.test_dir, self.raw_data_dir, self.raw_data_fn, 
                          'testing', device='cpu', processed_dir=processed_dir, 
                          **self.expt_kwargs)
        return expt

    def get_excluded_games(self, data_dir, split):
        exc_path = os.path.join(data_dir, f'processed/{split}_other_data.pkl')
        with open(exc_path, 'rb') as handle:
            other_data = pickle.load(handle)
            exc = other_data['excluded']
        exc_ids = [d.game_id for d in exc]
        return exc_ids

    def tear_down(self, data_dir, for_experiment_test=False):
        shutil.rmtree(os.path.join(data_dir, 'processed'), 
                      ignore_errors=True)
        shutil.rmtree(os.path.join(data_dir, 'checkpoints'), 
                      ignore_errors=True)
        if not for_experiment_test:
            if os.path.exists(os.path.join(data_dir, 'logger_info.pickle')):
                os.remove(os.path.join(data_dir, 'logger_info.pickle'))
            if os.path.exists(os.path.join(data_dir, 'split_inds.pkl')):
                os.remove(os.path.join(data_dir, 'split_inds.pkl'))


def continuous_to_discrete(data):
    # Input shopuld be EbbFlowGameData object
    t = np.arange(data.continuous['point_dir'].shape[0]) * data.step
    pt_ons, pt_offs, pt = _get_stimulus_bounds(data.continuous['point_dir'], 
                                               t, data.step)
    mv_ons, mv_offs, mv = _get_stimulus_bounds(data.continuous['mv_dir'], 
                                               t, data.step)
    cue_ons, cue_offs, cue = _get_stimulus_bounds(data.continuous['task_cue'], 
                                                  t, data.step)
    assert pt_ons == mv_ons == cue_ons
    assert pt_offs == mv_offs == cue_offs

    rts, resp_dirs = _get_resp_from_continuous(data.continuous['urespdir'], 
                                               t, pt_ons, pt_offs, data.step)

    discrete = {'onset': pt_ons,
                'offset': pt_offs,
                'point_dir': pt,
                'mv_dir': mv,
                'task_cue': cue,
                'urespdir': resp_dirs,
                'urt_samples': rts}

    return discrete


def _get_resp_from_continuous(d, t, onsets, offsets, step):
    resp_dirs = np.array([])
    rts = np.array([])
    onsets = np.array(onsets)
    unsorted_onsets = np.array([])
    for i in range(4):
        this_resp = d[:, i]
        this_peaks, _ = find_peaks(this_resp, height=0.5)
        this_abs_rts = t[this_peaks] / step
        this_abs_rts = this_abs_rts[this_abs_rts < offsets[-1]]
        for abs_rt in this_abs_rts:
            diffs = abs_rt - onsets
            diffs_pos = diffs[diffs > 0]
            ons_pos = onsets[diffs > 0]
            min_ind = np.argmin(diffs_pos)
            rts = np.append(rts, diffs_pos[min_ind])
            resp_dirs = np.append(resp_dirs, i)
            unsorted_onsets = np.append(unsorted_onsets, ons_pos[min_ind])

    sort_inds = np.argsort(unsorted_onsets)
    rts = rts[sort_inds].astype(int)
    resp_dirs = resp_dirs[sort_inds].astype(int)
    return rts.tolist(), resp_dirs.tolist()


def _get_stimulus_bounds(d, t, step):
    onsets = np.array([0])
    offsets = np.array([])
    first_dir = np.nonzero(d[0, :] == 1)[0][0]
    stim_dirs = np.array([first_dir])
    diff = d[1:, :] - d[:-1, :]
    for i in range(d.shape[1]):
        this_diff = diff[:, i]
        this_onsets = t[1:][this_diff == 1] / step
        this_offsets = t[1:][this_diff == -1] / step
        onsets = np.append(onsets, this_onsets)
        offsets = np.append(offsets, this_offsets)
        stim_dirs = np.append(stim_dirs, len(this_onsets) * [i])

    sort_inds = np.argsort(onsets)
    sort_ons = onsets[sort_inds].astype(int)
    sort_dirs = stim_dirs[sort_inds].astype(int)
    if len(onsets) - len(offsets) == 1:
        sort_ons = sort_ons[:-1]
        sort_dirs = sort_dirs[:-1]
    sort_offs = np.sort(offsets).astype(int)
    return sort_ons.tolist(), sort_offs.tolist(), sort_dirs.tolist()


def bin_response_times(rts, step):
    binned_rts = []
    for rt in rts:
        this_binned = step * np.rint(np.array(rt) / step)
        binned_rts.append(this_binned.tolist())
    return binned_rts


def get_raw_path(config):
    raw_dir = config['data_params']['data_path']
    raw_fn = config['data_params']['data_fn']
    return os.path.join(raw_dir, raw_fn)


def append_test_data(test_df_list, orig_data, test_game_id):
    this_test_df = orig_data.loc[orig_data['game_result_id'] == test_game_id]
    test_df_list.append(this_test_df)
    return test_df_list


def elbo_testing(model, xu, anneal_param):
    # Same as elbo objective, but doesn't average before returning; 
    # also returns inputs. This is a kluge to facilitate testing.
    x, px_w, w, _, w_means, w_vars = model(xu)
    log_like = (px_w.log_prob(x)).sum(0).sum(-1)
    lqw_x = model.qw_x(w_means, w_vars).log_prob(w).sum(0).sum(-1)
    lpw = model.pw(*model.pw_params).log_prob(w).sum(0).sum(-1)
    loss = anneal_param * log_like + anneal_param * lpw - lqw_x
    return -loss, x
