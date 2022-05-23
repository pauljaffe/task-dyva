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

Note: To rerun only a subset of the analyses, comment out the relevant code in make_paper.py. See the notes at the top of make_paper.py for additional options and info.


Quick start guide to model training
------------

To get started with training new models, do steps 1-4 above, then run the model training test script in examples/check_model_training from the command line:

```
poetry run python3 training_script.py
```

This is a toy example and will only run for a few epochs. To train a model with the same parameters as used in the paper, see examples/model_training_example/training_script.py. 

### Notes
1) While training on CPU is supported, training on GPU is strongly recommended. To toggle training on CPU vs. GPU, set the "device" parameter in the training script.

2) By default, model training runs will be logged using the Neptune logger (recommended; see https://docs.neptune.ai/getting-started/installation). To train without logging, simply set "do_logging" in "expt_kwargs" to False in the training script. 

3) To play around with the trained models analyzed in the paper, download the models linked above. The figure analysis files in /manuscript provide examples of how to do various analyses. 

4) To run all of the tests, including testing experiment logging, the Neptune logger needs to be configured. Run pytest with the --project_name argument set to the name of the Neptune project when running pytest:

```
poetry run pytest --project_name="my_project"
```

Only the tests in test_logging.py require the --project_name argument. 
