from definitions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.functions import *
from src.visualization.general import *


def missing_values_barplot(df, missing=True, return_vals=False):
    """ Takes a dataframe and plots the percentage of missing values (or not missing if set False) """
    # Sorted percentage of missing (or contained) values
    s_amount = 100 * df.isna().sum() / df.shape[0]
    if missing == False:
        s_amount = 100 - s_amount
    s_amount.sort_values(ascending=False, inplace=True)

    # Barplot it
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.barplot(s_amount.index, s_amount.values, ax=ax, palette="Blues_d")

    # Some niceties
    ax.set_title('% {} values'.format('Missing' if missing else 'Contained'), weight='bold', fontsize=16)
    plt.xticks(rotation=45)

    if return_vals:
        return s_amount


def plot_label_zero_to_one_ratio(labels, first_n=False, ax=None):
    """
    For each time point, computes the ratio of zeros to ones in the labels frame and outputs the ratios as a barplot.
    This allows analysis of how sepsis changes over time.

    :param labels: Binary labels dataframe.
    :param first_n: Set to int > 0 to plot only for the first n timepoints
    :return: ax object
    """
    # Ratio of zeros to ones at given times
    total_zeros = (1 - labels).groupby('time').apply(sum)
    total_ones = labels.groupby('time').apply(sum)
    ratios = total_ones / total_zeros

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(25, 10))

    # Plot
    if first_n is not False:
        sns.barplot(x=total_ones.index, y=ratios, order=total_ones.iloc[0:first_n].index, ax=ax)
    else:
        sns.barplot(x=total_ones.index, y=ratios, ax=ax)

    # Get rid of numerous ticks
    remove_plot_ticks(ax)

    return ax


def get_num_filled_entries_per_id(df):
    """ Gets the number of non na values for each column of each person """
    def func(df):
        not_na = (~df.isna()).sum(axis=0)
        return not_na.T
    non_na_sum = df.groupby('id').apply(func)
    return non_na_sum


if __name__ == '__main__':
    df = load_pickle(ROOT_DIR + '/data/interim/from_raw/df.pickle')
    a = get_num_filled_entries_per_id(df)
    save_pickle(a, ROOT_DIR + '/data/processed/transforms/num_filled_entries.pickle')
