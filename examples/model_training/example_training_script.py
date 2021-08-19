import os

from task_dyva import Experiment

# -----------------------------
# Example model training script
# -----------------------------

# Run this to make sure that the module is correclty installed.
# In practice, we would train for many more epochs, ideally on a GPU.
# The parameters assume that the Neptune logger has been configured.
# To train the model without logging, simply set 'do_logging' in
# expt_kwargs to False. 

# If successful, you should see a few files/folders in this directory
# at the conclusion of training: ...
# -----------------------------

# Some finagling to get the paths set
experiment_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(experiment_dir, 
                            '../../tests/test_data/user1365')
processed_data_dir = os.path.join(experiment_dir, 'processed_data')
raw_data_fn = 'user1365.pickle'
expt_name = 'example'
expt_tags = ['excellent', 'example']
device = 'cpu'

# Change a few parameters (instead of using the defaults in 
# config/model_config.yaml)
data_params = {'start_times': [5000], 'upscale_mult': 1}
expt_kwargs = {'do_logging': True, 'num_epochs': 5, 
               'nth_play_range': [150, 200], 'keep_every': 1,
               'train': data_params, 'val': data_params, 
               'test': data_params}

# Set up the experiment -- this will also process and save the data.
expt = Experiment(experiment_dir, raw_data_dir, raw_data_fn, expt_name, 
                  expt_tags=expt_tags, device=device, 
                  processed_dir=processed_data_dir, **expt_kwargs)

# Run a few training epochs
expt.run_training()
