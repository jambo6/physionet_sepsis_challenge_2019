"""
Other functions, not strictly for plotting but used in notebook visualization
"""
from definitions import *
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def get_num_ids(df):
    """ Returns the number of unique ids in a dataframe """
    return len(df.index.get_level_values('id').unique())


def num_id_remove(df, cols):
    """ For a given list of columns, returns the number of people who have column containing nan """
    num = get_num_ids(df[cols][df[cols].isna().sum(axis=1) > 0])
    print('For an entry for each of {} remove {} ids'.format(cols, num))
    return num


def get_scores(labels, predictions):
    """ Gets some standard scoring metrics """
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    return acc, f1, auc


def prediction_accuracy_at_different_timepoints(labels, predictions):
    """ Evaluation of prediction accuracy as we increase time """
    joint = pd.concat([labels, predictions], axis=1)
    joint.columns = ['labels', 'predictions']

    def func(df):
        labels, predictions = df['labels'].values, df['predictions'].values

        # Only do it if we have a sepsis case
        if (labels.sum() > 0) and ((1 - labels).sum() > 0):
            # Get the scores for the specific timepoints
            acc, f1, auc = get_scores(labels, predictions)
            cm = confusion_matrix(labels, predictions)

            # Add to a dataframe and return
            return pd.DataFrame(data=np.array([acc, f1, auc, cm]).reshape(1, -1), columns=['acc', 'f1', 'auc', 'cm'])

    # Get the scores for each timepoint
    scores = joint.groupby('time').apply(func)
    scores.index = scores.index.droplevel(-1)

    return scores


if __name__ == '__main__':
    # Get data
    results = load_pickle(ROOT_DIR + '/models/test/susceptible/results.pickle')

    # Expand it
    labels = results['labels']
    all_probas = results['all_probas'][0]
    final_probas = results['final_probas']

