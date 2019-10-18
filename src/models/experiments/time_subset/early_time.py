"""
Here we will run and experiment that will train an xgb model using data for each t < val for val in range(0, 58).

Results from the jupyter notebooks show us that
"""
# Set experiment
from sacred import Experiment
ex = Experiment('main_experiment')

# My imports
from definitions import *
from pprint import pprint
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMRegressor
from src.models.experiments.functions import *
from src.models.experiments.time_subset.functions import *
from src.models.functions import *
from src.features.transformers import *


@ex.config
def config():
    save_dir = None
    increase_utility = False
    increase_utility_method = 'basic'
    add_susceptible = False

    # Time subsetting options
    T = 1
    train_reduced = True

    # Extra features
    abs_sum_of_changes = False
    abs_changes_lookback = 10
    moments = False

    # Signatures
    columns = False
    irr_columns = False
    irr_lookback = 10
    irr_addtime = True
    irr_penoff = True

    # Classifier
    n_estimators = 100
    max_depth = 4
    learning_rate = 0.1
    num_leaves = 31

@ex.main
def run(_run, save_dir, increase_utility, increase_utility_method, add_susceptible,
        T, train_reduced,
        abs_sum_of_changes, abs_changes_lookback,
        moments, moment_lookback,
        columns,
        irr_columns, irr_lookback, irr_addtime, irr_penoff,
        n_estimators, max_depth, learning_rate, num_leaves
        ):
    # Add id to save dir
    _run.save_dir = save_dir + '/' + str(_run._id)

    # Load data
    df, labels_binary, labels_utility = load_munged_data()
    df.drop('hospital', axis=1, inplace=True)

    # Increase the utility function if specified
    if increase_utility is not False:
        labels_utility = IncreaseUtility(increase=increase_utility, method=increase_utility_method).transform(labels_utility)

    # If we are training on the reduced dataframe, reduce now to len T
    if train_reduced:
        df, labels_binary, labels_utility = reduce_data_to_T(df, labels_binary, labels_utility, T)

    # Add means and stds
    moment_frame = None
    if moments is not False:
        moment_frame = AddMoments(moments=moments, lookback=moment_lookback).transform(df)

    # Add signatures, do this before imputation
    signatures = None
    if columns is not False:
        signatures = add_signatures(df, columns)

    # Add irregular signatures
    # irr_signatures = None
    # if irr_columns is not False:
    #     irr_signatures = add_signatures(
    #         df, irr_columns, individual=True,
    #         lookback=irr_lookback, order=3, logsig=True, leadlag=True, addtime=irr_addtime, pen_off=irr_penoff
    #     )

    # Compile together
    data = np.concatenate([df, moment_frame, signatures], axis=1)
    df = pd.DataFrame(index=df.index, data=data)

    # Add probability of susceptible sepsis
    if add_susceptible:
        probas = load_pickle(MODELS_DIR + '/experiments/susceptibility/suscep/1/probas.pickle')
        df['susceptible_probas'] = probas.loc[df.index]

    # Perform cv
    cv = CustomStratifiedGroupKFold(n_splits=5).split(df, labels_binary)

    # Loop for probas
    clf = LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1)
    probas = cross_val_predict_to_series(clf, df, labels_utility.values, cv=cv, n_jobs=-1)
    ppprint('AUC: {:.3f}'.format(roc_auc_score(labels_binary, (probas > 0.5).astype(int))))

    # Loop for score, first predict ones in > T locations
    binary_preds, scores, _ = ThresholdOptimizer(labels=labels_binary, preds=probas).cross_val_threshold(cv, parallel=True, give_cv_num=True)
    ppprint('Mean early time score: {}'.format(np.mean(scores)))
    # preds, probas_full, scores, thresholds = perform_threshold_computation(probas, T, cv)

    # Make full data
    _, labels_binary, _ = load_munged_data()
    full_preds = pd.Series(index=labels_binary.index, data=1)
    full_preds.loc[binary_preds.index] = binary_preds

    # Get score
    score = ComputeNormalizedUtility().score(labels_binary, full_preds)

    _run.log_scalar('utility_score', np.mean(scores))
    # save_pickle(probas_full, _run.save_dir + '/probas.pickle')
    ppprint('T = {}, Score: {:.3f}\n'.format(T, score), color='green')



if __name__ == '__main__':
    # Some model setup
    experiment_name = input('Experiment name: ')
    ppprint("Running experiment '{}'".format(experiment_name), start=True)

    # Set save dir
    save_dir = create_fso(ex, experiment_name, dir=MODELS_DIR + '/experiments/time_subset')

    # Define the parameter grid
    param_grid = {
        'save_dir': [save_dir],
        'increase_utility': [False],
        'increase_utility_method': ['basic'],
        'T': [61],
        'train_reduced': [True],
        'add_susceptible': [False],

        # Extra features
        'abs_sum_of_changes': [False],
        'abs_changes_lookback': [10],
        'moments': [4],
        'moment_lookback': [7],

        # Signatures
        'columns': [
            ['SOFA', 'MAP', 'ShockIndex'],
            # ['SOFA']
        ],
        'irr_columns': [
            False,
            # ['FiO2', 'Glucose', 'SaO2'],
            # ['FiO2', 'SaO2', 'BaseExcess', 'Platelets', 'WBC', 'BUN'],
            # ['FiO2', 'Glucose'],
            # ['FiO2', 'SaO2']
        ],
        'irr_lookback': [15],
        'irr_addtime': [False],
        'irr_penoff': [False],

        'n_estimators': [150],
        'max_depth': [4],
        'learning_rate': [0.1]
        # 'num_leaves': [15, 25, 30, 40],
    }

    # Make the same every time
    pprint.pprint(param_grid)
    for i, params in enumerate(ParameterGrid(param_grid)):
        ppprint('Run number ' + str(i + 1) + '\n' + '-' * 50, color='magenta')
        print_experiment_params(param_grid, params)
        ex.run(config_updates=params)




