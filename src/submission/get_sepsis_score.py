#!/usr/bin/env python
from definitions import *
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from src.submission.generate_submission.main import *


def load_sepsis_model():
    loc = ROOT_DIR + '/models/submissions/submission_3'
    fname1 = loc + '/clf.pickle'
    fname3 = loc + '/threshold.pickle'

    models = {
        'clf': load_pickle(fname1),
        'threshold': load_pickle(fname3)
    }
    return models


def make_frame(data, column_names):
    """ Puts in the dataframe form that can be used by the algos """
    df = pd.DataFrame(data=data, columns=column_names)

    # Add the time index
    df.index = pd.MultiIndex.from_tuples((1, time) for time in range(df.shape[0]))
    df.index.names = ['id', 'time']

    return df


@timeit
def get_sepsis_score(data, models):
    # Open the models
    clf = models['clf']
    threshold = models['threshold']

    column_names = load_pickle(MODELS_DIR + '/other/col_names.pickle')
    column_names = column_names[0:40]

    # Make df (drop hospital col)
    df = make_frame(data, column_names)

    # Now process the data
    df = basic_data_process(df)
    df = process_data_and_build(df, config['feature'], submission=True)

    # Now make final prediction
    pred = clf.predict(df)

    # Turn into 0, 1
    pred = (pred > threshold).astype(int)

    return pred[0], pred[0]

