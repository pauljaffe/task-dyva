# Test logging with Neptune
# NOTE: the "project_name" argument has to be passed in from the 
# command line when running pytest. E.g. run:
#      >> poetry run pytest --project_name="my_project" ./test_logging.py
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

def test_logging(project_name, raw_data_dir=raw_data_dir, 
                 raw_data_fn=raw_data_fn, test_dir=test_dir):
    # Check that logging works correctly. Note: this creates a new run in 
    # Neptune - this can be deleted. 
    num_epochs = 3
    tparams = {'start_times': [5000], 'upscale_mult': 1, 
               'duration': 5000}
    expt_kwargs = {'mode': 'full', 'do_logging': True,
                   'do_early_stopping': False, 
                   'nth_play_range': [150, 200], 'train': tparams,
                   'val': tparams, 'test': tparams, 
                   'num_epochs': num_epochs, 'keep_every': 1,
                   'batch_size': 512}

    tester = SetUpTests(test_dir, raw_data_dir, raw_data_fn, 
                        project_name=project_name, **expt_kwargs)
    tester.tear_down(test_dir)
    expt = tester.make_experiment()

    # Check setting up new experiment
    expt.run_training()
    assert expt.iteration == num_epochs
