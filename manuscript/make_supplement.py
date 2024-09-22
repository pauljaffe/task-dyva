# Run the analyses and reproduce the figures from the paper
# (Extended Data only). See the README for detailed instructions.
import os
import time
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from figureS2 import FigureS2
from figureS3 import FigureS3
from figureS4 import FigureS4
from figureS5 import FigureS5
from figureS6 import FigureS6
from figureS7 import FigureS7
from figureS8 import FigureS8
from figureS9 import FigureS9
from figureS10 import FigureS10


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

figS2 = FigureS2(model_dir, figure_dir, metadata)
figS2.make_figure()

figS3 = FigureS3(model_dir, figure_dir, metadata, rand_seed, n_boot)
figS3.make_figure()

figS4 = FigureS4(model_dir, figure_dir, metadata, rand_seed, n_boot)
figS4.make_figure()

figS5 = FigureS5(model_dir, figure_dir, metadata, rand_seed, n_boot)
figS5.make_figure()

figS6 = FigureS6(model_dir, figure_dir, metadata)
figS6.make_figure()

figS7 = FigureS7(model_dir, figure_dir, metadata)
figS7.make_figure()

figS8 = FigureS8(model_dir, figure_dir, metadata, rand_seed, n_boot)
figS8.make_figure()

figS9 = FigureS9(model_dir, figure_dir, metadata)
figS9.make_figure()

figS10 = FigureS10(model_dir, figure_dir, metadata, rand_seed, n_boot)
figS10.make_figure()

run_time = time.time() - t0
print(f"Run time: {run_time}s")
