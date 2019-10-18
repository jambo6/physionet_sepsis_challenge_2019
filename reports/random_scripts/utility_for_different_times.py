"""
Gets the scores if we consider times only up to time T
"""
from definitions import *
import matplotlib.pyplot as plt
from src.models.optimizers import ThresholdOptimizer
from src.models.evaluators import ComputeNormalizedUtility

if __name__ == '__main__':
    # Load raw
    df = load_pickle(ROOT_DIR + '/data/interim/munged/df.pickle')
    labels_binary = load_pickle(ROOT_DIR + '/data/processed/labels/original.pickle')
    labels_utility = load_pickle(ROOT_DIR + '/data/processed/labels/utility_scores.pickle')
    labels = labels_binary.reset_index()
    ids_eventual = load_pickle(ROOT_DIR + '/data/processed/labels/ids_eventual.pickle')
    num_eventual = len(ids_eventual)

    # Drop time and mark
    labels_eventual = pd.Series(index=labels_binary.index.get_level_values('id').unique(), data=0)
    labels_eventual.loc[ids_eventual] = 1

    # Load preds
    probas = load_pickle(ROOT_DIR + '/models/experiments/main/finalised/1/probas.pickle')

    # Optimize
    thresh, score = ThresholdOptimizer(jupyter=True).optimize_utility(labels_binary, probas)

    # Make predictions
    preds = pd.Series(index=labels_binary.index, data=(probas > thresh).astype(int))

    # Find score for each time
    # tt = range(1, 20, 5)
    # scores = []
    # for t in tt:
    #     l, p = labels_binary.loc[pd.IndexSlice[:, :t]], preds.loc[pd.IndexSlice[:, :t]]
    #     scores.append(ComputeNormalizedUtility(jupyter=True).score(l, p))

    # Score for time ranges
    tt = [[0, 58], [58, 400]]
    scores = []
    for i in range(len(tt)):
        l, p = labels_binary.loc[pd.IndexSlice[:, tt[i][0]:tt[i][1]]], preds.loc[pd.IndexSlice[:, tt[i][0]:tt[i][1]]]

        # Get hospital mask
        hosp_mask = df['hospital'].loc[pd.IndexSlice[:, tt[i][0]:tt[i][1]]] == 1

        l1, p1 = l[hosp_mask], p[hosp_mask]
        l2, p2 = l[~hosp_mask], p[~hosp_mask]

        hosp_scores = []

        hosp_scores.append(ComputeNormalizedUtility(jupyter=True).score(l1, p1))
        hosp_scores.append(ComputeNormalizedUtility(jupyter=True).score(l2, p2))
        hosp_scores.append(ComputeNormalizedUtility(jupyter=True).score(l, p))
        scores.append(hosp_scores)


