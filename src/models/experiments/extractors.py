""" Methods to produce dataframe summaries and reports """
from definitions import *
import os
import pandas as pd


def load_runs(directory):
    # Dictionary with keys being the run name and containing config and metric dictionaries
    run_dict = {
        run_name: {
            'config': load_json(directory + '/' + run_name + '/config.json'),
            'metric': load_json(directory + '/' + run_name + '/metrics.json')
        }
        for run_name in os.listdir(directory) if not run_name.startswith('_')
    }
    return run_dict


def expand_config(config_dict):
    """
    Expands the nested config dict out into a single dict labelled by second params

    :param config_dict: The nested config dict
    :return: expanded dict
    """
    if 'seed' in config_dict.keys():
        del config_dict['seed']
    if 'save_dir' in config_dict.keys():
        del config_dict['save_dir']

    # If multilayered, expand the layers
    if any([isinstance(value, dict) for value in config_dict.values()]):
        config_dict = {key1 + '__' + key2: value2 for key1, value1 in config_dict.items() for key2, value2 in value1.items()}

    # Change all lists to strings else we get a problem
    config_dict = {key: str(value) for key, value in config_dict.items()}
    return config_dict


class RunToFrame():
    def __init__(self, get_data=True):
        self.get_data = get_data

    @staticmethod
    def get_dataframe(run, id):
        """ Gets the information that is to be put into a dataframe, does not include any metrics that start 'data.' """
        config, metrics = run['config'], run['metric']

        # Expand config
        config = expand_config(config)

        # Expand metrics
        metrics_expanded = {name: metrics[name]['values'][0] for name in metrics.keys()}

        # Make a dataframe of the config and append the one of the scores
        df_config = pd.DataFrame.from_dict(config, orient='index')
        df_metrics = pd.DataFrame.from_dict(metrics_expanded, orient='index')
        df_full = pd.concat([df_config, df_metrics]).T

        # Add the id
        df_full['id'] = id

        return df_full

    def transform(self, run_dir):
        run_dict = load_runs(run_dir)

        # Make the dataframe
        df_final = pd.concat([self.get_dataframe(run_entry, id) for id, run_entry in run_dict.items()], sort=True)
        df_final.set_index('id', inplace=True)

        # Make numeric when possible
        df_final = df_final.apply(pd.to_numeric, axis=1, errors='ignore')

        return df_final


def drop_unchanged_cols(df):
    """ Drop columns that have not changed in the run. """
    more_than_one_entry = (df.apply(pd.Series.nunique) != 1)
    keep_cols = more_than_one_entry[more_than_one_entry == True].index
    return df[keep_cols]


if __name__== '__main__':
    run_dir = MODELS_DIR + '/experiments/different_signature_features'
    # df = RunToFrame(get_data=True).transform(run_dir)
    # df = RunToFrame().transform(ROOT_DIR + '/models/experiments/13:01:41.dropping_potential_new_sig_features')
    # df = RunToFrame().transform(ROOT_DIR + '/models/experiments/early_time/2019-06-05/23:21:06.diff_sigs')

