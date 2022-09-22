# Run the analyses and reproduce the figures from the paper 
# (Extended Data only). See the README for detailed instructions.
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from manuscript.figureS2 import FigureS2
from manuscript.figureS3 import FigureS3
from manuscript.figureS4 import FigureS4
from manuscript.figureS5 import FigureS5
from manuscript.figureS6 import FigureS6
from manuscript.figureS7 import FigureS7
from manuscript.figureS8 import FigureS8
from manuscript.figureS9 import FigureS9
from manuscript.figureS10 import FigureS10


rand_seed = 12345
n_boot = 1000
fontsize = 5

# Specify paths to models/data, metadata, and a folder to save figures
model_dir = '/PATH/TO/MODELS/AND/DATA'  # change
expt_metadata_path = '/PATH/TO/METADATA'  # change
figure_dir = '/PATH/TO/SAVE/FOLDER'  # change
os.makedirs(figure_dir, exist_ok=True)
metadata = pd.read_csv(expt_metadata_path, header=0)

# Plotting params
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

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
print(f'Run time: {run_time}s')
