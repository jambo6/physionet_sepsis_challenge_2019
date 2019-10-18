PhysioNet 2019 Sepsis Challenge
==============================

The repository contains the code for our submission (Team Name: Can I get your Signature?) to the PhysioNet 2019 challenge. The code has not been edited since the final submission and as such is a little untidy, we will be working on making the codebase more userfriendly in future, please excuse the current mess!



Getting Started
---------------
Once the repo has been cloned locally, setup a python environment with ``python==3.6`` and run ``pip install -r requirements.txt``

you must download the official PhysioNet training data. This consists of two folders of .psv files. These folders should be placed in data/raw/training_{A, B}. 

To setup the data run ``src/data/make_dataframe.py``.

**Recommended** since the dataset is very large, for offline testing and experimentation, a test environement was created by placing a /test/ folder within data. The structure is then /data/test/{external, interim, processed, raw} that mimics the structure in /data/{external, interim, processed, raw}. This can be setup by running ``src/data/test_data.py`` after the data setup step. This will take a handful of septic cases, mix with some non-septic cases and generates save files in ``data/test`` that will be loaded if running in the test environemnt. 

To generate a model run ``src/models/experiments/main.py`` and input a name for the experiment, this will end up being the name of the folder in ``models/`` that will contain run information. Inside this file is a dictionary of options that can be edited in model training. It includes choices of what features to compute and any associated hyperparameters. For example changing:
~~~
{ ...
    'feature__columns': ['SBP', 'DBP', 'HR'],
    'feature__order': [5, 6],
... }
~~~
will run two experiments, the first computing signatures of Systolic BP, Diastolic BP and HR to order 5 the second to order 6, with other options as specified. 

Once run, this will save model metrics and the probabilities output for each timestamp in ``models/{test, if in test env}/experiment_name/run_num/``. Here experiment_name is the input specified on running ``src/models/experiments/main.py`` and ``run_num`` is a number that increases by 1 for every option combination. In the example above the order 5 computation is run number 1 and the order 6 is run number 2. 

To use the full dataset rather than experiment in the test environment the paths in ``definitions.py`` must be changed:
~~~
# From
# Setup paths
DATA_DIR = ROOT_DIR + '/data/test'
MODELS_DIR = ROOT_DIR + '/models/test'

# To
# Setup paths
DATA_DIR = ROOT_DIR + '/data'
MODELS_DIR = ROOT_DIR + '/models'
~~~
I had a simple if statement that chose between depending on whether I was working locally or via ssh. 



Project Organization
--------------------

    ├── Sakefile           <- Sakefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Papers that may be useful for the project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── signatures 
    │   │   |    └── compute_signatures.py 
    │   │   |    └── signature_functions.py 
    │   │   |    └── transformers.py 
    │   │   └── build_features.py   
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── omni           <- Generic functions used everywhere, such as load_pickle
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── environment.yml    <- Environment setup file
    |
    └── definitions.py     <- File to load useful global info such as ROOT_DIR


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
