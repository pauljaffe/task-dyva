import os
import argparse

from task_dyva import Experiment

# This CLI trains a task-DyVA model on data specified in the command-line arguments.
# By default, the model will be trained using the parameters in model_config.yaml.

parser = argparse.ArgumentParser()
parser.add_argument(
    "raw_data_dir", 
    help=(
        "Path to directory with raw data used to train the model, "
        "e.g. $MODEL_DIR/ages30to39_u3531_expt1 if downloaded from Zenodo."
    ),
)
parser.add_argument(
    "save_dir",
    help=(
        "Parent directory to save checkpoints from training runs. "
        "Checkpoints for this run will be saved in save_dir/expt_name."
    ),
)
parser.add_argument(
    "expt_name",
    help=(
        "Name for this training run/experiment. "
        "Checkpoints for this run will be saved in save_dir/expt_name."
    ),
)
parser.add_argument(
    "-d",
    "--device",
    nargs="?",
    type=str,
    help="Device to run training on, e.g. 'cpu' or 'cuda:0'",
)
parser.add_argument(
    "-np",
    "--neptune",
    nargs="?",
    type=str,
    help=(
        "Project name for logging training metrics with Neptune."
        "Not compatible with -tb flag (TensorBoard logging)."
    ),
)
parser.add_argument(
    "-tb",
    "--tensorboard",
    nargs="?",
    type=str,
    help=(
        "Directory under save_dir to save TensorBoard log files."
        "Not compatible with -np flag (Neptune logging)."
    ),
)

args = parser.parse_args()
raw_data_dir = args.raw_data_dir
raw_data_fn = "data_pre_split.pickle"
expt_save_dir = os.path.join(args.save_dir, args.expt_name)
processed_data_dir = os.path.join(expt_save_dir, "processed_data")
expt_name = args.expt_name
device = args.device

assert args.neptune is None or args.tensorboard is None, (
        "Only one of TensorBoard or Neptune logging option can be set!"
        )

if args.neptune is not None:
    expt_kwargs = {'logger_type': 'neptune',
                   'neptune_proj_name': args.neptune}
elif args.tensorboard is not None:
    expt_kwargs = {'log_save_dir': os.path.join(args.save_dir, args.tensorboard)}

expt = Experiment(expt_save_dir, raw_data_dir, raw_data_fn, expt_name, 
                  device=device, processed_dir=processed_data_dir, 
                  **expt_kwargs)
expt.run_training()
