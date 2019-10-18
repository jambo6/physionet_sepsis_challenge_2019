from definitions import *
import multiprocessing as mp
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_predict
# from xgboost import XGBClassifier, XGBRegressor
num_workers = mp.cpu_count()


class ComputeNormalizedUtility():
    """
    A version of the normalized utility scoring function that can be applied to series as opposed to folders of files

    NOTE: This requires there to exist an ids_eventual.pickle file that contains the ids for those people that
    eventually develop sepsis.
    """
    def __init__(self, threshold=None, jupyter=False):
        self.threshold = threshold
        # Use normal location if in havok
        self.scores_loc = DATA_DIR + '/processed/labels/full_scores.pickle' if jupyter is False else ROOT_DIR + '/data/processed/labels/full_scores.pickle'

    @staticmethod
    def compute_normalized(actual_score, inaction_score, perfect_score):
        # Give the normalised utility function
        return (actual_score - inaction_score) / (perfect_score - inaction_score)

    def get_scores(self, s, p):
        """ Gets the perfect, inaction and actual scores. """
        perfect_score = s[:, [0, 1]].max(axis=1).sum()
        inaction_score = s[:, 0].sum()
        actual_score = self.actual_score(s, p)
        return actual_score, inaction_score, perfect_score

    @staticmethod
    def actual_score(s, p):
        """ Gets the actual score through numpy arrays for a speeed up. """
        p = p.squeeze()
        actual_score = s[:, 1][p == 1].sum() + s[:, 0][p == 0].sum()
        return actual_score

    def score(self, labels, predictions, cv_num=False):
        # Apply thresh if specified
        if self.threshold is not None:
            predictions = apply_threshold(predictions, threshold=self.threshold)

        if cv_num is not False:
            # Try to load from save else compute and save
            save_loc = DATA_DIR + '/processed/labels/scores/cv_{}'.format(cv_num)
            try:
                # If the exist, load them in
                scores = load_pickle(save_loc + '/scores.pickle')
                inaction_score = load_pickle(save_loc + '/inaction_score.pickle')
                perfect_score = load_pickle(save_loc + '/perfect_score.pickle')
                actual_score = self.actual_score(scores.values, predictions.values)
            except:
                # Load scores and save things that we can precompute
                scores = load_pickle(self.scores_loc)
                scores = scores.loc[predictions.index]
                actual_score, inaction_score, perfect_score = self.get_scores(scores.values, predictions.values)

                # Save so we dont have to recompute these every time in the full data case
                save_pickle(scores, save_loc + '/scores.pickle')
                save_pickle(inaction_score, save_loc + '/inaction_score.pickle')
                save_pickle(perfect_score, save_loc + '/perfect_score.pickle')
        else:
            scores = load_pickle(self.scores_loc)
            scores = scores.loc[labels.index]
            actual_score, inaction_score, perfect_score = self.get_scores(scores.values, predictions.values)

        # Compute utility
        utility = self.compute_normalized(actual_score, inaction_score, perfect_score)

        return utility


@numpy_method
def apply_threshold(predictions, threshold=0.5):
    """ Converts the predictions object to a binary one according to the threshold value """
    if isinstance(threshold, np.ndarray): threshold = threshold[0]
    prediction_values = np.array([1 if x > threshold else 0 for x in predictions]).reshape(-1)
    return prediction_values


def basic_xgb_predict(X, y, regression=False):
    """ Simple xgb classifier for general analysis """
    xgb = XGBClassifier() if not regression else XGBRegressor()
    xgb.fit(X, y)
    predictions = cross_val_predict(xgb, X, y, cv=3)
    return predictions


def get_metrics(labels, predictions, print_metrics=True):
    """ Gets some basic metrics on the predictions and stores in a dict """
    # Setup dict
    metrics = {}

    # Store metrics
    metrics['acc'] = accuracy_score(labels, predictions)
    metrics['auc'] = roc_auc_score(labels, predictions)
    metrics['f1'] = f1_score(labels, predictions)
    metrics['cm'] = confusion_matrix(labels, predictions)

    # Print if specified
    if print_metrics:
        _print_metrics(metrics)

    return metrics


def score_at_single_time(labels, probas, t):
    """ For a given time t, returns the utility score at that time. """
    mask = (labels['time'] == t)
    labels, probas = labels[mask].set_index(['id', 'time']), probas[mask].set_index(['id', 'time'])
    score = ComputeNormalizedUtility(jupyter=(not HAVOK)).score(labels, probas)
    return score


def score_at_each_time(labels, probas, T):
    """
    Returns a list of the utility scores at each time values.

    :param labels (pd.Series): Binary labels.
    :param probas (pd.Series): Prediction probas.
    :param T (int): Time to go up to in the evaluation.
    :return scores: List of scores where scores[i] is the score at time i.
    """
    # Setup cv loop
    args = [(labels.reset_index(), probas.reset_index(), t) for t in range(T)]
    scores = basic_parallel_loop(score_at_single_time, args, parallel=True)
    return scores


def _print_metrics(metrics):
    print('\nMetrics for the run:')
    print('ACC: {:.2f}%, AUC: {:.2f}, F1: {:.2f}'.format(metrics['acc'], metrics['auc'], metrics['f1']))
    print(metrics['cm'])


if __name__ == '__main__':
    # Get data and probas
    df = load_pickle(ROOT_DIR + '/data/interim/munged/df.pickle')
    labels = load_pickle(ROOT_DIR + '/data/processed/labels/original.pickle')
    probas = load_pickle(ROOT_DIR + '/models/experiments/main/plspls/1/probas.pickle')

    # We want to mark labels as zero unless SOFA deterioration is > 2.
    from src.models.optimizers import ThresholdOptimizer
    thresh, score = ThresholdOptimizer(jupyter=True, budget=200).optimize_utility(labels, probas)
    print(score)

    # Now add a new feature, cumsum of deterioration.
    det_cumsum = df['SOFA_deterioration'].groupby('id').apply(lambda x: x.cumsum())

    # If at places where the cumsum is < 2, mark the proba as 0
    from copy import deepcopy
    probas_new = deepcopy(probas)
    det_cumsum.loc[pd.IndexSlice[:, 61:]] = 10
    probas_new.loc[pd.IndexSlice[:, 61:]] = 10
    probas_new.loc[det_cumsum < 1] = -100

    thresh_new, score_new = ThresholdOptimizer(budget=200).optimize_utility(labels, probas_new)
    print(score_new)






