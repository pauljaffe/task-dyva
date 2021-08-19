# Test trial resampling operations used in taskdataset module
import os

import numpy as np

from task_dyva import EbbFlowStats
from task_dyva.testing import SetUpTests


this_dir = os.path.dirname(os.path.abspath(__file__))
rel_data_dir = 'user1365'
raw_data_dir = os.path.join(this_dir, 'test_data', rel_data_dir)
raw_data_fn = 'user1365.pickle'
test_dir = raw_data_dir


def test_resampling(raw_data_dir=raw_data_dir, raw_data_fn=raw_data_fn, 
                    test_dir=test_dir):
    # Check that resampled data has similar statistics to original data
    rel_tol = 0.05
    resamp_types = [None, 'kde']
    rt_means = []

    tparams = {'smoothing_type': 'gaussian',
               'noise_sd': 0, 'rt_method': 'max',
               'start_times': [5000, 55000],
               'upscale_mult': 5, 'duration': 5000}

    expt_kwargs = {'outlier_method': 'mad', 'outlier_thresh': 10,
                   'mode': 'full', 'do_logging': False, 
                   'nth_play_range': [150, 200], 'train': tparams}

    for rs in resamp_types:
        expt_kwargs['train']['data_augmentation_type'] = rs
        tester = SetUpTests(test_dir, raw_data_dir, raw_data_fn, **expt_kwargs)
        # Remove any processed data etc. from previous runs
        tester.tear_down(test_dir)

        expt = tester.make_experiment()
        data = expt.train_dataset
        rates = data.xu[:, :, :4]
        stats_obj = EbbFlowStats(rates, data)
        stats = stats_obj.get_stats()
        rt_means.append(stats['u_mean_rt'])

    assert np.allclose(rt_means[0], rt_means[1], rtol=rel_tol)
