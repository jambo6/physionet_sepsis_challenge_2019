from definitions import *
import scipy.stats as stats


def get_eventual_sepsis_labels(drop_positive_utility=False):
    """

    :param save:
    :param drop_positive_utility:
    :return:
    """
    LABELS_DIR = DATA_DIR + '/processed/labels'
    try:
        if drop_positive_utility:
            labels = load_pickle(LABELS_DIR + '/eventual_sepsis_utility_dropped.pickle')
        else:
            labels = load_pickle(LABELS_DIR + '/eventual_sepsis.pickle')
    except:
        # Get those that eventually develop sepsis
        ids_eventual = load_pickle(LABELS_DIR + '/ids_eventual.pickle')
        labels = load_pickle(LABELS_DIR + '/utility_scores.pickle')

        # If we are dropping cases that dont have a utility score = -0.05. This is to understand if there is information
        # in the dataframe before the labelled times
        if drop_positive_utility:
            labels = labels[labels == -0.05].astype(int)

        # Set all as zero, the set eventual sepsis cases as 1
        labels = pd.Series(index=labels.index, data=0)
        labels.loc[ids_eventual] = 1

        if drop_positive_utility:
            save_pickle(labels, LABELS_DIR + '/eventual_sepsis_utility_dropped.pickle')
        else:
            save_pickle(labels, LABELS_DIR + '/eventual_sepsis.pickle')
    return labels


def get_eventual_sepsis_utilities(return_df=True):
    """ Gets the utility labels of those who eventually develop sepsis. """
    # Get the labels and reduce to only those with eventual sepsis
    labels = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')
    sepsis_ids = load_pickle(DATA_DIR + '/processed/labels/ids_eventual.pickle')
    labels = labels.loc[sepsis_ids]

    # Now reduce the dataframe
    if return_df:
        df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
        df = df.loc[labels.index]
        return df, labels
    else:
        return labels


def to_tsfresh_form(df, labels):
    """ Puts the data into a form where tsfresh extractors can be used. """
    tsfresh_frame = df.reset_index()
    labels = labels.groupby('id').apply(lambda x: x.iloc[0]).astype(int)
    return tsfresh_frame, labels


def first_n_timepoints(df, labels, n=30):
    """ Finds people who have at least n timepoints and returns entries up to that timepoint """
    # Ids of those who have n entries
    at_least_n_id = df.loc[pd.IndexSlice[:, n], :].index.get_level_values('id').unique()

    # Then get the first n entries of those people
    df_first_n = df.loc[pd.IndexSlice[at_least_n_id, :n], :]

    # Do to the labels
    labels = labels.loc[df_first_n.index]

    return df_first_n, labels


def get_long_time_sepsis_cases(df, labels_eventual, window_size=10, gap=5):
    """
    Gets 'long time' sepsis cases. That is, the sepsis cases of people in the icu for t > 58. It then splits into a
    'near sepsis' chunk, which is the data from onset time (t=-3 from end) and a chunk before that (separated by 'gap').
    The goal is to understand the processes at play that differ between the hours before sepsis and the hours preceeding
    this.

    :param df (dataframe): Full dataframe.
    :param labels_eventual (series): Labels as 0 if person doesnt end up with sepsis or 1 everywhere if they do.
    :param window_size (int): The size of the window with which to consider
    :param gap (int): Gap between the pos and the neg cases
    :return (tuple): (data, labels). Data contains the pos and neg timeseries, the neg ones will have negative indexes.
    The labels will be one for the end series and 0 for the early times.
    """
    # Get the > 58 cases that survive 58 hrs
    long_time_ids = df.loc[pd.IndexSlice[:, 60], :].index.get_level_values('id')
    labels_single = labels_eventual.groupby('id').apply(lambda x: x.iloc[0])
    long_time_sepsis_ids = [x for x in labels_single[labels_single == 1].index if x in long_time_ids]

    # Now reduce the df to the > 58 cases that get sepsis
    df = df.loc[long_time_sepsis_ids]

    # Now we wish to split into at sepsis and before sepsis to understand the differences
    window_size = 10
    gap = 30
    at_sepsis = df.groupby('id', as_index=False).apply(lambda x: x.iloc[-window_size:])
    away_from = df.groupby('id', as_index=False).apply(lambda x: x.iloc[-2*window_size-gap:-window_size-gap])

    # Drop None idxs
    at_sepsis.index = at_sepsis.index.droplevel(None)
    away_from.index = away_from.index.droplevel(None)

    # Make the away from ids negative to have unique indexes
    away_from.reset_index(inplace=True)
    away_from['id'] = -away_from['id']
    away_from.set_index(['id', 'time'], inplace=True)

    # Make compiled dataframe
    data = pd.concat([at_sepsis, away_from])

    # Finally make the labels
    labels_away = pd.Series(index=away_from.index, data=0)
    labels_sepsis = pd.Series(index=at_sepsis.index, data=1)
    labels = pd.concat([labels_away, labels_sepsis])

    return data, labels


