from definitions import *
from copy import deepcopy
import matplotlib.pyplot as plt
from src.models.functions import load_munged_data

# Load data
df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
_, labels_binary, labels_utility = load_munged_data()
ids_eventual = load_pickle(DATA_DIR + '/processed/labels/ids_eventual.pickle')

# We want to see temperature sampling effects on sepsis.
temp = deepcopy(df['Temp'])
temp[~temp.isna()] = 1
temp.fillna(0, inplace=True)

# Make a variable for 24 hrs before sepsis
labels_utility_new = deepcopy(labels_utility)
labels_utility_new.groupby('id').apply()

# Check if labels increase near the point of sepsis
temp[labels_utility >= 0].sum() / temp[labels_utility >= 0].shape[0]
temp[labels_utility < 0].sum() / temp[labels_utility < 0].shape[0]

