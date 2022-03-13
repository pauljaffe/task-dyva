Task-DyVA
------------

Task-DyVA (pronounced dee-vuh) is a framework for modeling sequential response time data from cognitive tasks. This repo contains code to train models and reproduce the analyses from the paper. 

Preprint: ### 

Trained models & data: ###


Reproduce the figures and analyses from the paper
------------

The easiest and recommended way to reproduce the results from the paper is as follows:

1) Fork the repo from the command line:

```
git clone https://github.com/pauljaffe/task-dyva
```

2) Install Poetry to manage the dependencies: https://python-poetry.org/docs/

3) Install the task-DyVA dependencies: Run `poetry install` from the command line within the local task-dyva repo. The complete set of dependencies is listed in pyproject.toml.

4) Download the models, data, and metadata: ###

5) Change the paths in make_paper.py to the local copies of the models/data, metadata, and a folder to save the figures. 

6) Run the script to make the figures and reproduce the analyses:

```
poetry run python3 ./make_paper.py
```

Note: To rerun only a subset of the analyses, comment out the relevant code in make_paper.py. See the notes at the top of make_paper.py for additional options and info.


Quick start guide to model training
------------

To get started with training new models, do steps 1-4 above, then run the model training test script in examples/check_model_training from the command line:

```
poetry run python3 ./training_script.py
```

This is a toy example and will only run for a few epochs. To train a model with the same parameters as used in the paper, see examples/model_training_example/training_script.py. 

### Notes
1) While training on CPU is supported, training on GPU is strongly recommended. To toggle training on CPU vs. GPU, set the "device" parameter in the training script.

2) By default, model training runs will be logged using the Neptune logger (recommended; see https://docs.neptune.ai/getting-started/installation). To train without logging, simply set "do_logging" in "expt_kwargs" to False in the training script. 

3) To play around with the trained models analyzed in the paper, download the models linked above. The figure analysis files in /manuscript provide examples of how to do various analyses. 

4) To run the tests, run the following from the command line:

```
poetry run pytest
```

Note that in order for the tests in test_experiment.py to work properly, the Neptune logger needs to be configured, and the NEPTUNE_PROJECT environment variable must be set.
