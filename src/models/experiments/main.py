from sacred import Experiment
from ingredients import feature_ingredient, generate_features
from ingredients import train_ingredient, train_model
ex = Experiment('main_experiment', ingredients=[feature_ingredient, train_ingredient])

# Other imports
from definitions import *
import pprint
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from src.models.experiments.functions import *
from src.data.extracts import irregular_cols, cts_cols, drop_cols, derived_cols


# Main run function
@ex.main
def run(_run, save_dir):
    # Give the run object the save directory in case we want to save there
    _run.save_dir = save_dir + '/' + str(_run._id)

    # Run subfunctions
    generate_features()
    train_model()


if __name__ == '__main__':
    # Setup
    experiment_name = input('Experiment name: ')
    ppprint("Running experiment '{}'".format(experiment_name), start=True)

    # Set save dir
    save_dir = create_fso(ex, experiment_name, dir=MODELS_DIR + '/experiments/main')

    # Define the parameter grid
    param_grid = {
        # Feature selection
        'feature__feature_selection': [False],

        # Additional features
        'feature__num_measurements': [False],

        # Moments
        'feature__moments': [False],
        'feature__moment_lookback': [7],

        # Signature options
        'feature__individual': [True],
        'feature__lookback': [7],
        'feature__lookback_method': ['fixed'],
        'feature__columns': [
            # False,
            ['SOFA', 'BUN/CR', 'MAP'],
        ],
        'feature__order': [3],
        'feature__logsig': [True],
        'feature__leadlag': [True],
        'feature__addtime': [False],

        # Cumsum signatures
        # 'feature__cs_columns': [False],
        'feature__cs_columns': [irregular_cols + cts_cols],
        'feature__cs_lookback': [7],
        'feature__cs_order': [3],
        'feature__cs_logsig': [True],
        'feature__cs_leadlag': [True],
        'feature__cs_addtime': [False],

        # Other
        'feature__extra_features': [True],
        'feature__add_max': [True],
        'feature__add_min': [True],
        'feature__max_min_lookback': [6],

        'feature__drop_count': [False],
        'feature__drop_count_moments': [False],
        'feature__drop_specific': [
            False,
        ],
        'feature__drop_specific_all': [
            False,
        ],

        # LGBM options
        'train__cv_hospital': [True],
        'train__binary': [False],
        'train__gs_params': [False],
        'train__n_estimators': [100],
        'train__learning_rate': [0.1],
    }

    # Make the same every time
    pprint.pprint(param_grid)
    for i, params in enumerate(ParameterGrid(param_grid)):
        ppprint('Run number ' + str(i + 1) + '\n' + '-'*50, color='magenta')
        print_experiment_params(param_grid, params)
        param_updater(params, feature_ingredient, train_ingredient)
        ex.run(config_updates={'save_dir': save_dir})



