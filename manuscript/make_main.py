# Run the analyses and reproduce the figures from the paper
# (main figures only). See the README for detailed instructions.
import os
import time
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import Preprocess
from figure2 import Figure2
from figure3 import Figure3
from figure4 import Figure4
from figure5 import Figure5


# Plotting and analysis params
batch_size = 512
rand_seed = 12345
n_boot = 1000
fontsize = 5
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_dir",
    help=(
        "Directory with all model subdirectories (e.g. ages20to29_u1076_expt1). "
        "and metadata.csv"
    ),
)
parser.add_argument(
    "-f",
    "--figure_dir",
    default="figures",
    nargs="?",
    type=str,
    help="Name of directory within model_dir to save figures (defaults to 'figures')",
)
parser.add_argument(
    "-p",
    "--do-preprocessing",
    action="store_true",
    help="Recompute all model outputs (not necessary to reproduce the figures)",
)
args = parser.parse_args()

model_dir = args.model_dir
figure_dir = os.path.join(model_dir, args.figure_dir)
os.makedirs(figure_dir, exist_ok=True)
metadata = pd.read_csv(os.path.join(model_dir, "model_metadata.csv"), header=0)

# Run summary analyses and create figures
t0 = time.time()
if args.do_preprocessing:
    preprocessing = Preprocess(
        model_dir, metadata, rand_seed, batch_size=batch_size
    )
    preprocessing.run_preprocessing()

fig2 = Figure2(model_dir, figure_dir, metadata, rand_seed, n_boot)
fig2.make_figure()

fig3 = Figure3(model_dir, figure_dir, metadata)
fig3.make_figure()

fig4 = Figure4(model_dir, figure_dir, metadata, rand_seed, n_boot)
fig4.make_figure()

fig5 = Figure5(model_dir, figure_dir, metadata, rand_seed, n_boot)
fig5.make_figure()

run_time = time.time() - t0
print(f"Run time: {run_time}s")
