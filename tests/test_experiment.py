# Test Experiment class
import os
import copy
import time

import pytest

from task_dyva.testing import SetUpTests


this_dir = os.path.dirname(os.path.abspath(__file__))
rel_data_dir = 'user1365'
raw_data_dir = os.path.join(this_dir, 'test_data', rel_data_dir)
raw_data_fn = 'user1365.pickle'
test_dir = raw_data_dir


@pytest.mark.parametrize('do_log, do_stop, delete_local, \
                          orig_training_epochs, \
                          resume_training_epochs, orig_true_epochs, \
                          resume_true_epochs, patience', [
    (True, True, False, 5, 5, 3, 4, 2),
    (False, False, False, 3, 6, 3, 6, 0),
    (True, False, True, 3, 6, 3, 6, 0)
])
def test_experiment(do_log, do_stop, delete_local, orig_training_epochs,
                    resume_training_epochs, orig_true_epochs,
                    resume_true_epochs, patience, raw_data_dir=raw_data_dir,
                    raw_data_fn=raw_data_fn, test_dir=test_dir):
    # Check that logging, resuming training, and early stopping
    # work correctly. Note: this creates a few runs in Neptune --
    # these can be deleted. 

    tparams = {'start_times': [5000], 'upscale_mult': 1, 
               'duration': 5000}
    expt_kwargs = {'mode': 'full', 'do_logging': do_log,
                   'do_early_stopping': do_stop, 
                   'nth_play_range': [150, 200], 'train': tparams,
                   'val': tparams, 'test': tparams, 
                   'num_epochs': orig_training_epochs, 'keep_every': 1,
                   'stop_patience': patience, 'stop_min_epoch': 0,
                   'stop_delta': float('inf'), 'stop_metric': 'switch_con_avg',
                   'batch_size': 512}

    tester = SetUpTests(test_dir, raw_data_dir, raw_data_fn, **expt_kwargs)
    tester.tear_down(test_dir)
    expt = tester.make_experiment()

    # Check setting up new experiment
    expt.run_training()
    assert expt.iteration == orig_true_epochs

    # First ensure that enough time has elapsed for info to upload
    # to Neptune
    time.sleep(30)
    # Check resuming training
    if delete_local:
        tester.tear_down(test_dir, for_experiment_test=True)
    resume_expt_kwargs = copy.deepcopy(expt_kwargs)
    resume_expt_kwargs['num_epochs'] = resume_training_epochs
    tester = SetUpTests(test_dir, raw_data_dir, raw_data_fn, 
                        **resume_expt_kwargs)
    expt = tester.make_experiment()
    expt.run_training()
    assert expt.iteration == resume_true_epochs
