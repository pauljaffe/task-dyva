# Run the analyses and reproduce the figures from the paper
# (main figures only). See the README for detailed instructions.
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from manuscript.preprocessing import Preprocess
from manuscript.figure2 import Figure2
from manuscript.figure3 import Figure3
from manuscript.figure4 import Figure4
from manuscript.figure5 import Figure5


do_preprocessing = False
batch_size = 512
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
if do_preprocessing:
    preprocessing = Preprocess(model_dir, metadata, rand_seed, 
                               batch_size=batch_size)
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
print(f'Run time: {run_time}s')
