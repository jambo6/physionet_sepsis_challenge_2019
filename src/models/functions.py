"""
Basic functions used in models sections
"""
from definitions import *
from multiprocessing import Pool
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_predict
from src.features.transformers import SignaturesFromIdDataframe, MakePaths
from src.features.signatures.transformers import SignatureTransformer, RemoveNanRows, AddTime
from src.features.transformers import FeatureSaverMixin


def load_munged_data(DATA_DIR=DATA_DIR):
    """ Loads the munged dataframe and the binary and utility labels """
    df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
    labels_binary = load_pickle(DATA_DIR + '/processed/labels/original.pickle')
    labels_utility = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')
    return df, labels_binary, labels_utility


@timeit
def cross_val_predict_to_series(clf, df, labels, groups=None, cv=5, method='predict', n_jobs=5):
    """ Performs cross val predict and returns the preidctions as a pandas series. """
    # Predict
    pred = cross_val_predict(clf, df, labels, groups=groups, cv=cv, method=method, n_jobs=n_jobs)

    if method == 'predict_proba':
        pred = pred[:, 1]

    # Make series
    pred_series = pd.Series(index=df.index, data=pred)
    return pred_series


@timeit
def parallel_cv_loop(func, cv, give_cv_num=False, parallel=True):
    """
    Performs a parallel training loop over the cv train_idx and test_idxs.

    Example:
        - func will usually be a class that contains df, labels info but __call__ method will run a single training loop
        given train_idx, test_idx
        - This will run func.__call__(train_idx, test_idx) for each idx pair in cv and return results
    :param func: Class that has information relating to df, labels and takes a __call__(train_idx, test_idx) to run loop
    :param cv: List of [(train_idx, test_idx), ...] pairs
    :param give_cv_num: Gives the cv num to the underlying function, used when using the full dataset and loading
    precomputed arrays for a specific cv_num
    :param parallel: Set to false for a for loop (allows for debugging)
    :return: results. A list of whatever func outputs for each cv idxs
    """
    if give_cv_num:
        cv = [(train_idx, test_idx, cv_num) for cv_num, (train_idx, test_idx) in enumerate(cv)]

    if parallel:
        pool = Pool(len(cv))
        results = pool.starmap(
            func, cv
        )
        pool.close()

    # Oldschool
    else:
        results = []
        for args in cv:
            results.append(func(*args))

    return results


class CustomStratifiedGroupKFold(TransformerMixin, BaseEstimator):
    """
    Picks folds with approx equal number of septic cases across folds.
    """
    def __init__(self, n_splits=5, seed=3):
        self.n_splits = n_splits
        self.seed = seed if isinstance(seed, int) else np.random.randint(10000)


    def split(self, data, labels, groups=False):
        # Set a seed so the same cv is used everytime
        np.random.seed(self.seed)

        # For grouping
        if groups is False:
            groups = data.index.get_level_values('id')

        # Split the ids into n pieces
        one_ids = np.array(list(labels[labels == 1].groupby('id').sum().index))
        np.random.shuffle(one_ids)

        # Get the one ids for the validation set
        val_ones = np.array_split(one_ids, self.n_splits)

        # Train ones
        train_ones = [list(set(one_ids) - set(x)) for x in val_ones]

        # Now cv the training set
        zero_ids = [x for x in np.unique(groups) if x not in one_ids]
        np.random.shuffle(zero_ids)
        val_zeros = np.array_split(np.array(zero_ids), self.n_splits)
        train_zeros = [list(set(zero_ids) - set(x)) for x in val_zeros]

        # Compile together
        id_groups = [(list(train_zeros[i]) + list(train_ones[i]), list(val_zeros[i]) + list(val_ones[i]))
                     for i in range(self.n_splits)]

        # Finally, get the indexes
        cv_iter = [(np.argwhere(np.isin(groups, x[0]) == True).reshape(-1),
                    np.argwhere(np.isin(groups, x[1]) == True).reshape(-1))
                   for x in id_groups]

        return cv_iter


@timeit
def add_signatures(df, columns, individual=True, lookback=7, lookback_method='fixed',
                   order=3, logsig=True, leadlag=True, addtime=False, cumsum=False, pen_off=False, append_zero=False,
                   last_only=False):
    """
    Method to extend a dataframe with signatures of specified columns. Allows specification to compute the signatures
    for each column individually
    """
    # Set the options
    options = {
        'order': order, 'logsig': logsig, 'leadlag': leadlag, 'add_time': addtime,
        'cumsum': cumsum, 'pen_off': pen_off, 'append_zero': append_zero,
        'use_esig': True
    }

    # Compute the signatures
    if individual:
        signatures = []
        for column in columns:
            # If col is in irregular cols, use addtime which will remove the nan values
            get_signatures = SignaturesFromIdDataframe(columns=[column], lookback=lookback,
                                                       lookback_method=lookback_method, options=options,
                                                       last_only=last_only)
            signatures.append(get_signatures.transform(df))
        signatures = np.concatenate(signatures, axis=1)
    else:
        get_signatures = SignaturesFromIdDataframe(columns=columns, lookback=lookback,
                                                   lookback_method=lookback_method, options=options, last_only=last_only
                                                   )
        signatures = get_signatures.transform(df)

    return signatures


class IncreaseUtility(FeatureSaverMixin):
    """ Increases the utility score of the eventual sepsis labels. """
    def __init__(self, increase=0.1, method='basic'):
        """
        :param labels_utility:
        :param increase (float > 0): The amount to increase by
        :param method (str): The method with which to perform the increase. So far 'linear' and 'basic' where 'basic' is a
        simple addition to all points, 'linear' creates a linear increase from 0 to first util > -0.05, then adds on the
        increase to the rest.
            'lookback' - Will increase the utility in a lookback window around the value.
        :return: labels_utility_increased
        """
        self.increase = increase
        self.method = method

        # Save loc for feature saver
        self.save_loc = DATA_DIR + '/processed/labels/increaed_utility/method={}_increase={}.pickle'.format(increase, method)

    def linear_increase(self, s):
        # Make a linear interpolation of the neg vals
        neg_vals = s[s == -0.05]

        # First add to the normal values
        s.loc[s[s > -0.05].index] += self.increase

        # Then create the linspace
        if neg_vals.shape[0] > 0:
            s.loc[neg_vals.index] = np.linspace(-0.05, self.increase, neg_vals.shape[0])

        return s

    def lookback_increase(self, s, lookback=10):
        """ Increase only in a lookback window around the point. """
        neg_vals = s[s == -0.05]

        # First add to the normal values
        s.loc[s[s > -0.05].index] += self.increase

        # Then add to the lookback window
        if neg_vals.shape[0] > 0:
            idxs = neg_vals.iloc[-lookback:].index
            s.loc[idxs] += self.increase

        return s

    @timeit
    def transform_func(self, labels_utility):
        # Get the sepsis locations
        sepsis_ids = load_pickle(DATA_DIR + '/processed/labels/ids_eventual.pickle')

        if self.method == 'basic':
            labels_utility.loc[sepsis_ids] += self.increase
        elif self.method == 'linear':
            lin_inc = labels_utility.loc[sepsis_ids].groupby('id').apply(self.linear_increase)
            labels_utility.loc[sepsis_ids] = lin_inc
        elif self.method == 'lookback':
            lb_inc = groupby_apply_parallel(labels_utility.loc[sepsis_ids].groupby('id'), self.lookback_increase)
            # lb_inc = labels_utility.loc[sepsis_ids].groupby('id').apply(self.lookback_increase)
            labels_utility.loc[sepsis_ids] = lb_inc

        return labels_utility


def remove_useless_columns(df):
    """
    Removes repeated columns from a dataframe or columns that contain the same value (these often come out of the signature).
    """
    # Remove same valued cols
    data, idx = np.unique(df.values, axis=1, return_index=True)
    columns = df.columns[idx]

    # Remake the df to use pandas functions to drop duplicated cols
    df = pd.DataFrame(index=df.index, data=data, columns=columns)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


@timeit
def numpy_to_named_dataframe(data, idx, name, last_only=False):
    """ Turns a numpy array into a dataframe with name_i for the ith col. """
    col_names = [name + '_{}'.format(i) for i in range(1, data.shape[-1] + 1)]
    if last_only is False:
        if isinstance(data, pd.DataFrame): data = data.values
        return pd.DataFrame(index=idx, data=data, columns=col_names)
    else:
        if isinstance(data, pd.DataFrame): data=data.values
        return pd.DataFrame(index=idx, data=data[[-1], :], columns=col_names)
