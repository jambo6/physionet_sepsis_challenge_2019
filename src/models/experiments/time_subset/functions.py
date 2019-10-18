from definitions import *
from xgboost import XGBRegressor
from src.models.functions import *
from src.models.optimizers import ThresholdOptimizer
from src.models.evaluators import ComputeNormalizedUtility

#
# def add_signatures(df, columns, individual, options=None):
#     # Set the options
#     if options is None:
#         options = {
#             'order': 2, 'logsig': True, 'leadlag': True,
#         }
#
#     # Compute the signatures
#     if columns is not False:
#         if individual:
#             for column in columns:
#                 add_signatures = SignaturesFromIdDataframe(columns=[column], lookback=7,
#                                                            lookback_method='fixed', options=options)
#                 df = add_signatures.transform(df)
#         else:
#             add_signatures = SignaturesFromIdDataframe(columns=columns, lookback=7,
#                                                        lookback_method='fixed', options=options)
#             df = add_signatures.transform(df)
#
#     return df


def reduce_data_to_T(df, labels_binary, labels_utility, T):
    """ Reduces the data to contain 0 < t < T if early=True else to T <= t < np.inf"""
    idx_slice = pd.IndexSlice[:, :T]

    # Reduce df
    df = df.loc[idx_slice, :]

    # Reduce labels in accordance with df
    labels_binary, labels_utility = labels_binary.loc[df.index], labels_utility.loc[df.index]

    return df, labels_binary, labels_utility


def perform_threshold_computation(probas, T, cv):
    # Get the binary labels
    labels_binary = load_pickle(DATA_DIR + '/processed/labels/original.pickle')

    # Make a full prediction frame of 1s
    probas_full = pd.Series(index=labels_binary.index, data=1)

    # Then fill the positions up to T with the probas
    probas_full.loc[pd.IndexSlice[:, :T]] = probas.loc[pd.IndexSlice[:, :T]]

    # Now perform thresholding
    preds, scores, thresholds = ThresholdOptimizer(labels_binary, probas_full).cross_val_threshold(cv, give_cv_num=True, parallel=True)

    return preds, probas_full, scores, thresholds


