import os

from task_dyva import Experiment

# -----------------------------
# Example model training script
# -----------------------------
# Run this script to train a model using the same parameters as used 
# in the paper. Set "raw_data_dir" to specify the location of the gameplay data 
# to be used for model training. 

# To run the model training, run:
# poetry run python3 ./example_training_script.py
# from the command line.

# Notes
# -----
# 1) To test that everything is correctly configured, run the test
# model training script in examples/test_model_training. 

# 2) While training on CPU is supported, training on GPU is strongly 
# recommended (toggle by setting the "device" parameter below).

# 3) The settings below assume that the Neptune logger has been 
# configured (recommended). To train without logging, simply set "do_logging"
# in "expt_kwargs" to False.
# Install Neptune: https://docs.neptune.ai/getting-started/installation


# Some finagling to get the paths set
experiment_dir = os.path.dirname(os.path.abspath(__file__))
#raw_data_dir = '/PATH/TO/DATA'  # change
raw_data_dir = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/models_data/test'
raw_data_fn = 'data_pre_split.pickle'
processed_data_dir = os.path.join(experiment_dir, 'processed_data')
expt_name = 'name_your_experiment'
expt_tags = ['add_some_tags']
device = 'cpu'  # For GPU, set to e.g. 'cuda:0'
expt_kwargs = {'do_logging': True}

# Set up the experiment: this will also process and save the processed data
expt = Experiment(experiment_dir, raw_data_dir, raw_data_fn, expt_name, 
                  expt_tags=expt_tags, device=device, 
                  processed_dir=processed_data_dir, **expt_kwargs)

# Run the training loop
expt.run_training()
