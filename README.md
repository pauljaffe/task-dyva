Task-DyVA
------------

Task-DyVA (pronounced dee-vuh) is a framework for modeling sequential response time data from cognitive tasks. This repo contains code to train models and reproduce the analyses from the paper. 

Preprint: Jaffe PI, Poldrack RA, Schafer RJ, Bissett PG. 2022. Discovering dynamical models of human behavior. bioRxiv. 

Trained models & data: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6368413.svg)](https://doi.org/10.5281/zenodo.6368413)


Reproduce the figures and analyses from the paper
------------

The easiest and recommended way to reproduce the results from the paper is as follows:

1) Fork the repo from the command line:

```
git clone https://github.com/pauljaffe/task-dyva
```

2) Install Poetry to manage the dependencies (see https://python-poetry.org/docs/). After installing, make sure that Poetry's bin directory is in the 'PATH' environment variable by running `source $HOME/.poetry/env` from the command line. 

3) Install the task-DyVA dependencies: Run `poetry install` from the command line within the local task-dyva repo. The complete set of dependencies is listed in pyproject.toml.

4) Download the models, data, and metadata linked above. 

5) Change the paths in make_paper.py to the local copies of the models/data, metadata, and a folder to save the figures. 

6) Run the script to make the figures and reproduce the analyses from the top-level directory of the task-DyVA repo:

```
poetry run python3 make_paper.py
```

### Notes 
1) To rerun only a subset of the analyses, comment out the relevant code in make_paper.py. See the notes at the top of make_paper.py for additional options and info.

2) To play around with the trained models analyzed in the paper, download the models linked above. The figure analysis files in /manuscript provide examples of how to do various analyses. 


Quick start guide to model training
------------

To get started with training new models, do steps 1-4 above, then run the model training test script in examples/check_model_training from the command line:

```
poetry run python3 training_script.py
```

### Notes
1) This is a toy example and will only run for a few epochs. To train a model with the same parameters as used in the paper, see examples/model_training_example/training_script.py. 

2) By default, these example scripts use TensorBoard for logging training metrics. See "Tracking model training" below for instructions on how to customize experiment tracking.

3) While training on CPU is supported, training on GPU is strongly recommended. To toggle training on CPU vs. GPU, set the "device" parameter in the training script.

4) To run the tests, run the following from the command line:

```
poetry run pytest
```

5) Here are example training curves from a successful run:
![image not found](successful_training.png "successful run")
The x-axis of each plot corresponds to the training epoch. The entire run is shown up until early stopping was triggered. The upper left plot shows the progression of the loss on the validation set over the course of training. The other three plots track the progression of the model's mean RT, switch cost, and congruency effect relative to the participant (see "Tracking model training" below for a description of the variables that are tracked during training). <br> 

Occasionally, the loss will diverge and training will ultimately fail. This appears to result from instabilities in the latent dynamical system (e.g. exponential growth), rather than exploding gradients (since gradient clipping is used). This can be diagnosed by examining the loss, which exhibits a sudden and dramatic increase (see below). The model's behavioral metrics also typically diverge concurrently. One easy fix is to use a different random seed to initialize training. To do so, simply set the 'rand_seed' key word argument in Experiment. 
![image not found](failed_training.png "failed run")

Tracking model training
------------

- Describe logged variables

