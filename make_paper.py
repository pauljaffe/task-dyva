# Run the analyses and reproduce the figures from the manuscript
import os

import pandas as pd
import matplotlib.pyplot as plt

from manuscript.preprocessing import Preprocess
from manuscript.figure2 import Figure2
from manuscript.figure3 import Figure3
from manuscript.figureS2 import FigureS2
from manuscript.figureS3 import FigureS3
from manuscript.figureS4 import FigureS4
from manuscript.figureS5 import FigureS5
from manuscript.figureS6 import FigureS6


# ADD NOTES
reload_primary_outputs = True
reload_behavior_summary = True
reload_fixed_points = True
reload_lda_summary = True
model_dir = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/models_data'
expt_metadata_path = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/model_metadata.csv'
metadata = pd.read_csv(expt_metadata_path, header=0)

# Plotting params
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.labelsize'] = 5
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

# SAVING (add notes)
figure_dir = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/initial_submission/figures'
os.makedirs(figure_dir, exist_ok=True)

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

fig3 = Figure3(model_dir, figure_dir, metadata)
fig3.make_figure()

# SUPPLEMENTAL FIGURES
#figS2 = FigureS2(model_dir, figure_dir, metadata)
#figS2.make_figure()

#figS3 = FigureS3(model_dir, figure_dir, metadata)
#figS3.make_figure()

#figS4 = FigureS4(model_dir, figure_dir, metadata)
#figS4.make_figure()

#figS5 = FigureS5(model_dir, figure_dir, metadata)
#figS5.make_figure()

#figS6 = FigureS6(model_dir, figure_dir, metadata)
#figS6.make_figure()
