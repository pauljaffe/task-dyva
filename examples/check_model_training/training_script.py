import os

from task_dyva import Experiment

# -------------------
# Test model training
# -------------------
# Run this script to ensure that everything is correctly configured
# (this only runs for a few epochs). If the training was successful, 
# a few files/folders will be created in the directory containing this script: 
# checkpoints, processed_data, and some metadata.

# To run the model training, run:
# poetry run python3 training_script.py
# from the command line.

# Notes
# -----
# 1) This is a toy example and will not yield a fully-trained model. 
# To train an actual model, see the script in examples/model_training_example.

# 2) To run the tests multiple times, delete all of the files/folders
# in the folder with the training script (other than the training script itself).


# Some finagling to get the paths set
experiment_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(experiment_dir, 
                            '../../tests/test_data/user1365')
processed_data_dir = os.path.join(experiment_dir, 'processed_data')
raw_data_fn = 'user1365.pickle'
expt_name = 'example'
device = 'cpu'

# Change a few parameters for testing (instead of using the defaults in 
# config/model_config.yaml)
data_params = {'start_times': [5000], 'upscale_mult': 1}
expt_kwargs = {'num_epochs': 5, 'nth_play_range': [150, 200], 
               'keep_every': 1, 'train': data_params, 'val': data_params, 
               'test': data_params}

# Set up the experiment: this will also process and save the processed data
expt = Experiment(experiment_dir, raw_data_dir, raw_data_fn, expt_name, 
                  device=device, processed_dir=processed_data_dir, 
                  **expt_kwargs)

# Run the training loop
expt.run_training()
