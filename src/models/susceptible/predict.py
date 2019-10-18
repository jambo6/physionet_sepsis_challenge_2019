"""
Predicting whether someone is going to get sepsis or not from the output of predicting_susceptibility.py
"""
from definitions import *
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from src.models.functions import load_munged_data
from src.models.optimizers import ThresholdOptimizer
from src.models.evaluators import ComputeNormalizedUtility, score_at_each_time


# Load the raw data
df, labels_binary, labels_utility = load_munged_data()
labels_eventual = labels_binary.groupby('id').apply(max)

# Suceptible probas
probas = load_pickle(MODELS_DIR + '/experiments/susceptibility/suscep/1/probas.pickle')

# First, get the score at each time for the best threshold
t, s = ThresholdOptimizer().optimize_utility(labels_binary, probas)
# t_scores = score_at_each_time(labels_binary, (probas > t).astype(int), 20)

# Now choose a thresh, set to 1 forever if t > thresh from the point of exceeding
# thresh = 0.15
# ids_thresh = probas[probas > thresh].index.get_level_values('id').unique()
# def make_one_from_thresh(probas, thresh):
#     # Setup vals to return
#     return_vals = np.zeros(shape=probas.shape)
#
#     # Index of threshold exceeding
#     first_arg = np.argwhere(probas.values > thresh)
#
#     # Make anything after (and including) the time of first exceed to be 1
#     if first_arg.shape[0] != 0:
#         return_vals[first_arg[0][0]:] = 1
#
#     return pd.Series(index=probas.index, data=return_vals)
# predictions = groupby_apply_parallel(probas.groupby('id'), make_one_from_thresh, 0.22)
# susceptible_score = ComputeNormalizedUtility().score(labels_binary, predictions)
# ppprint('Score using susceptible method: {:.3f}'.format(susceptible_score))

# Get max's
max_probas = probas.groupby('id').apply(max)

# Print confusion matrix
thresh = 0.17
ids = labels_binary.loc[pd.IndexSlice[:, 15:]].index.get_level_values('id').unique()
print('ROC: {:.3f}'.format(roc_auc_score(labels_eventual.loc[ids], max_probas.loc[ids])))
cm = confusion_matrix(labels_eventual.loc[ids], (max_probas.loc[ids] > thresh).astype(int))
print(cm)

# Predict 0-1
thresh = 0.14
exceeds = probas[probas > thresh].reset_index()['id'].value_counts()
id_exceeds = exceeds[exceeds >= 3].index
predictions = pd.Series(index=labels_eventual.index, data=0)
predictions.loc[id_exceeds] = 1
cm = confusion_matrix(labels_eventual, predictions)
print(cm)



