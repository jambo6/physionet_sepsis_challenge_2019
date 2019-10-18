# This must be run with havok
from definitions import *
from src.data.transformers import LabelsToScores

# Get all data
df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
labels = load_pickle(DATA_DIR + '/processed/labels/original.pickle')

# Get the ids that have some 1s in
sepsis_ids = labels[labels.isin((1,))].index.get_level_values('id').unique()
other_ids = [x for x in df.index.get_level_values('id').unique() if x not in sepsis_ids]

# Lets use 1000 total
ids = list(sepsis_ids[0:50]) + other_ids[0:100]

save_pickle(df.loc[ids], DATA_DIR + '/test/interim/munged/df.pickle')
save_pickle(labels.loc[ids], DATA_DIR + '/test/processed/labels/original.pickle')
scores = LabelsToScores().transform(labels.loc[ids])
save_pickle(scores, DATA_DIR + '/test/processed/labels/utility_scores.pickle')

