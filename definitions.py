import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set true if submission
SUBMISSION = False

# HAVOK to note if we are in ssh
HAVOK = False

# Warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # LightGBM warning


# Setup paths
DATA_DIR = ROOT_DIR + '/data/test'
MODELS_DIR = ROOT_DIR + '/models/test'
# else:
#     HAVOK = True
#     DATA_DIR = '/scratch/morrill/physionet2019/data'
#     MODELS_DIR = ROOT_DIR + '/models'


# Packages/functions used everywhere
from src.omni.decorators import *
from src.omni.functions import *
from src.omni.base import *