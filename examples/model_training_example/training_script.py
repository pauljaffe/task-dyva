import os

from task_dyva import Experiment

# -----------------------------
# Example model training script
# -----------------------------
# Run this script to train a model using the same parameters as used 
# in the paper. Set "raw_data_dir" to specify the location of the gameplay data 
# to be used for model training. 

# To run the model training, activate the task-dyva conda environment and run:
# >> python training_script.py
# from the command line.

# Notes
# -----
# 1) To test that everything is correctly configured, run the test
# model training script in examples/check_model_training. 

# 2) While training on CPU is supported, training on GPU is strongly 
# recommended (toggle by setting the "device" parameter below).

# 3) By default, TensorBoard is used for experiment tracking. Set the
# 'log_save_dir' keyword argument to specify where the logging data will
# be saved. To track the experiment with Neptune, set the 'logger_type'
# kwarg to 'neptune' and set the 'neptune_proj_name' kwarg to the name
# of your project as it is stored in Neptune. See the README for more guidance
# on how to configure experiment tracking. 


# Some finagling to get the paths set
experiment_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = '/path/to/data'  # change
raw_data_fn = 'data_pre_split.pickle'
processed_data_dir = os.path.join(experiment_dir, 'processed_data')
expt_name = 'name_your_experiment' # change
device = 'cpu'  # For GPU, set to e.g. 'cuda:0'
expt_kwargs = {'log_save_dir': '/path/to/tensorboard/log/dir'} # change

# To use Neptune, set the following variables:
#expt_kwargs = {'logger_type': 'neptune',
#               'neptune_proj_name': '/my/neptune/project', 
#               'expt_tags': ['tag1', 'tag2']}

# Set up the experiment: this will also process and save the processed data
expt = Experiment(experiment_dir, raw_data_dir, raw_data_fn, expt_name, 
                  device=device, processed_dir=processed_data_dir, 
                  **expt_kwargs)

# Run the training loop
expt.run_training()
