from definitions import *
import shutil
import pprint
from datetime import datetime
from multiprocessing import Pool
from sacred.observers import FileStorageObserver


def create_fso(ex, experiment_name, dir=MODELS_DIR + '/experiments'):
    """
    Creates a save location under dir/data/time.experiment_name. dir defaults to experiments but can be change to
    experiments/subfolder for example
    :param ex: The experiment being run
    :param experiment_name: The name of the experiment
    :param dir: (optional) save dir location
    :return: directory name
    """
    # Add the date and time for clearer model labelling
    # datetime_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # date, time = datetime_string.split(' ')

    # Create the save_dir
    directory = dir + '/' + experiment_name

    # Create the experiment FSO
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    ex.observers.append(FileStorageObserver.create(directory))

    return directory


def get_type_params(params, type):
    """
    Extracts the parameters of a specific type when the format is given like with sklearn pipelines. That is:
        params = {
            'type1__param_name': [params1],
            'type2__param_name': [params2]
        }
    """
    type_params = {
        __name.split('__')[1]: param_list for __name, param_list in params.items() if __name.split('__')[0] == type
    }
    return type_params


def param_updater(params, feature_ingredient, train_ingredient):
    """ Updates various configuration parameters """
    feature_params = get_type_params(params, 'feature')
    if feature_params != {}:
        feature_ingredient.add_config(**feature_params)

    train_params = get_type_params(params, 'train')
    if train_params != {}:
        train_ingredient.add_config(**train_params)


def print_experiment_params(param_grid, params):
    """ Prints the params that are being run of those that are being changed in the experiment. """
    changing_keys = [key for key, value in param_grid.items() if len(value) > 1]
    changing_params = {key: value for key, value in params.items() if key in changing_keys}
    pprint.pprint(changing_params)




