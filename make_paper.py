# Run the analyses and reproduce the figures from the manuscript
import os

import pandas as pd
import matplotlib.pyplot as plt

from manuscript.preprocessing import Preprocess
from manuscript.figure2 import Figure2
from manuscript.figureS2 import FigureS2
from manuscript.figureS3 import FigureS3
from manuscript.figureS4 import FigureS4
from manuscript.figureS5 import FigureS5


# Plotting params
plt.rcParams['svg.fonttype'] = 'none'

# Set rerurn_preprocessing to True to recompute the model outputs and
# run other preliminary analyses (this takes a while to run).
rerun_preprocessing = False
model_dir = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/models_data'
expt_metadata_path = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/model_metadata.csv'
metadata = pd.read_csv(expt_metadata_path, header=0)

# SAVING (add notes)
figure_dir = '/Users/paul.jaffe/Dropbox/Manuscripts/task-DyVA/initial_submission/figures'
os.makedirs(figure_dir, exist_ok=True)


# Rerun the analyses and make the figures
if rerun_preprocessing:
    preprocessing = Preprocess(model_dir, metadata, 
                               load_saved_analysis=False)
    preprocessing.run_preprocessing()

# MAIN FIGURES
#fig2 = Figure2(model_dir, figure_dir, metadata)
#fig2.make_figure()

# SUPPLEMENTAL FIGURES
#figS2 = FigureS2(model_dir, figure_dir, metadata)
#figS2.make_figure()

#figS3 = FigureS3(model_dir, figure_dir, metadata)
#figS3.make_figure()

#figS4 = FigureS4(model_dir, figure_dir, metadata)
#figS4.make_figure()

figS5 = FigureS5(model_dir, figure_dir, metadata)
figS5.make_figure()
