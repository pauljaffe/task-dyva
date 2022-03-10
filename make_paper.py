# Run the analyses and reproduce the figures from the manuscript
import os

import pandas as pd
import matplotlib.pyplot as plt

from manuscript.preprocessing import Preprocess
from manuscript.figure2 import Figure2
from manuscript.figure3 import Figure3
from manuscript.figure4 import Figure4
from manuscript.figure5 import Figure5
from manuscript.figureS2 import FigureS2
from manuscript.figureS3 import FigureS3
from manuscript.figureS4 import FigureS4
from manuscript.figureS5 import FigureS5
from manuscript.figureS6 import FigureS6
from manuscript.figureS7 import FigureS7
from manuscript.figureS8 import FigureS8
from manuscript.figureS9 import FigureS9


# Notes
# -------------
# 1) To recreate the figures and calculate summary statistics, 
# set the paths to the models, metadata, and save folder below.
# Then run the command: poetry run python3 ./make_paper.py 
# in a terminal window (poetry must be installed; see README). 

# 2) To rerun all preprocessing, set the four "reload_#" variables to False.
# The full preprocessing pipeline takes a long time to run
# (~12 hours on a 2017 MacBook Pro w/ a 2.8GHz processor & 16Gb RAM).

# 3) To recreate the figures and calculate summary statistics using
# precomputed preprocessing outputs, set the four "reload_#" variables to True
# (this will take several minutes to run). 

# 4) A subset of the figures can be recreated by commenting out the code below.


# Toggle reloading precomputed preprocessing outputs
reload_primary_outputs = True
reload_behavior_summary = True
reload_fixed_points = True
reload_lda_summary = True

# Specify paths to models, metadata, and a folder to save figures
model_dir = '/PATH/TO/MODELS'
expt_metadata_path = '/PATH/TO/METADATA'
figure_dir = '/PATH/TO/SAVE/FOLDER'
os.makedirs(figure_dir, exist_ok=True)
metadata = pd.read_csv(expt_metadata_path, header=0)

# Plotting params
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.labelsize'] = 5
plt.rcParams['axes.titlesize'] = 5
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

# Run preprocessing
preprocessing = Preprocess(model_dir, metadata, 
                           reload_primary_outputs=reload_primary_outputs,
                           reload_behavior_summary=reload_behavior_summary,
                           reload_fixed_points=reload_fixed_points,
                           reload_lda_summary=reload_lda_summary)
preprocessing.run_preprocessing()

# MAIN FIGURES
#fig2 = Figure2(model_dir, figure_dir, metadata)
#fig2.make_figure()

#fig3 = Figure3(model_dir, figure_dir, metadata)
#fig3.make_figure()

fig4 = Figure4(model_dir, figure_dir, metadata)
fig4.make_figure()

fig5 = Figure5(model_dir, figure_dir, metadata)
fig5.make_figure()

# SUPPLEMENTAL FIGURES
figS2 = FigureS2(model_dir, figure_dir, metadata)
figS2.make_figure()

figS3 = FigureS3(model_dir, figure_dir, metadata)
figS3.make_figure()

figS4 = FigureS4(model_dir, figure_dir, metadata)
figS4.make_figure()

figS5 = FigureS5(model_dir, figure_dir, metadata)
figS5.make_figure()

figS6 = FigureS6(model_dir, figure_dir, metadata)
figS6.make_figure()

figS7 = FigureS7(model_dir, figure_dir, metadata)
figS7.make_figure()

figS8 = FigureS8(model_dir, figure_dir, metadata)
figS8.make_figure()

figS9 = FigureS9(model_dir, figure_dir, metadata)
figS9.make_figure()
