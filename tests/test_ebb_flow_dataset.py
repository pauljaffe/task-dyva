# Tests for EbbFlowDataset, also tests the transform SmoothResponses
import os
import copy

import pytest

from task_dyva import EbbFlowStats
from task_dyva.testing import SetUpTests, bin_response_times, \
    continuous_to_discrete

this_dir = os.path.dirname(os.path.abspath(__file__))
rel_data_dirs = ['user1365', 'user1089', 'user2139']
raw_data_dirs = [os.path.join(this_dir, 'test_data', rdd)
                 for rdd in rel_data_dirs]
raw_data_fns = ['user1365.pickle',
                'user1089.pickle',
                'user2139.pickle']
test_dirs = raw_data_dirs
test_conditions = [
    (None, 'kde', 0.1, 'center_of_mass', None, None),
    (None, 'gaussian', 0, 'max', 'mad', 10),
    ('kde', 'adaptive_gaussian', 0, 'max', 'mad', 10)
]


@pytest.mark.parametrize(
    'resamp, sm, noise_sd, rt_method, out_method, out_thresh', 
    test_conditions
)
def test_ebb_flow_dataset(resamp, sm, noise_sd, rt_method, out_method, 
                          out_thresh, raw_data_dirs=raw_data_dirs, 
                          raw_data_fns=raw_data_fns, test_dirs=test_dirs):

    discrete_check_fields = ['onset', 'offset', 'point_dir', 'mv_dir', 
                             'task_cue', 'urespdir', 'urt_samples']
    tparams = {'data_augmentation_type': resamp, 'smoothing_type': sm,
               'noise_sd': noise_sd, 'rt_method': rt_method,
               'start_times': [5000, 20000, 35000, 50000],
               'upscale_mult': 2, 'duration': 5000}
    train_p, val_p = copy.deepcopy(tparams), copy.deepcopy(tparams)
    test_p = copy.deepcopy(tparams)
    test_p['upscale_mult'] = 1
    test_p['duration'] = 10000
    expt_kwargs = {'outlier_method': out_method, 'outlier_thresh': out_thresh,
                   'mode': 'full', 'nth_play_range': [150, 200], 
                   'train': train_p, 'val': val_p, 'test': test_p}
    splits = ['train', 'val', 'test']

    for rdd, rfn, td in zip(raw_data_dirs, raw_data_fns, test_dirs):
        tester = SetUpTests(td, rdd, rfn, **expt_kwargs)
        # Remove any processed data etc. from previous runs
        tester.tear_down(td)

        expt = tester.make_experiment()
        split_datasets = [expt.train_dataset, expt.val_dataset, 
                          expt.test_dataset]
        pre_datasets = tester.get_pre_datasets(expt)

        for dataset, pre, split in zip(split_datasets, pre_datasets, splits):
            params = dataset.params

            # Ensure all games are accounted for across processed + excluded
            pre_ids = pre['game_result_id']
            processed_ids = dataset.game_ids
            exc_ids = tester.get_excluded_games(td, split)
            assert set(pre_ids) == set(processed_ids + exc_ids)

            # Ensure the number of sub-game trial sequences is correct
            num_starts = len(params['start_times'])
            upscale_mult = params['upscale_mult']
            n_check = len(set(pre_ids)) * num_starts * upscale_mult
            n_actual = len(processed_ids) + len(exc_ids)
            assert n_check == n_actual

            # Check shape of model inputs
            if split in ['train', 'val']:
                assert dataset.xu.shape[0] == 250
            else:
                assert dataset.xu.shape[0] == 500
            assert dataset.xu.shape[2] == 14

            # Check that discrete -> continuous mapping is correct
            if resamp is None and sm == 'gaussian':
                for g in range(len(dataset)):
                    gdata = dataset.get_processed_sample(g)
                    gdisc = gdata.discrete
                    gdisc2disc = continuous_to_discrete(gdata)
                    for f in discrete_check_fields:
                        assert gdisc[f] == gdisc2disc[f]

            rates = dataset.xu[:, :, :4]
            binned_rts = bin_response_times(dataset.discrete['urt_ms'], 
                                            dataset.params['step_size'])
            dataset.discrete['urt_ms'] = binned_rts
            stats_obj = EbbFlowStats(rates, dataset)
            stats = stats_obj.get_stats()

            # Test EbbFlowStats: ensure that when we use the user's 
            # smoothed responses for the "model's" responses, stats computed 
            # from user and "model" are the same.
            if resamp is None and sm == 'gaussian':
                assert stats['u_switch_cost'] == stats['m_switch_cost']
                assert stats['u_con_effect'] == stats['m_con_effect']
