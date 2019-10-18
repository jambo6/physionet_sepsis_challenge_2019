""" Attempting to understand further the causes of sepsis vs the non causes """
from sacred import Experiment
ex = Experiment('susceptibility_experiment')

# My imports
from definitions import *
from pprint import pprint
from sklearn.model_selection import ParameterGrid, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBRegressor
from src.features.transformers import AddMoments, MostRecentChange, AbsoluteSumOfChanges
from src.models.functions import *
from src.models.susceptible.functions import *
from src.models.experiments.functions import create_fso
from src.data.extracts import drop_cols, irregular_cols, cts_cols


# Setup experiment
@ex.config
def config():
    drop_cols = False
    moments = False
    moment_lookback = 10
    moment_irr_lookback = 10
    last_change = False
    abs_sum_of_changes = False
    abs_changes_lookback = 10
    get_baseline = False
    baseline_shift = 6
    baseline_roll = 5

    # Signatures
    sig_columns = False
    sig_order = 2
    sig_lookback = 6
    sig_penoff = False
    sig_addtime = False

    # Irr signatures
    irr_sig_columns = False
    irr_sig_order = 2
    irr_sig_lookback = 6
    irr_sig_penoff = False
    irr_sig_addtime = False


def abs_sum_func(df):
    abs_changes = (df.shift(1) - df).abs()
    abs_sum = abs_changes.rolling(10).sum()
    return abs_sum

def ae_func(df):
    abs_energy = df.rolling(10).apply(lambda x: (x ** 2).sum())
    return abs_energy

@ex.main
def run(_run, save_dir, drop_cols, moments, moment_lookback, moment_irr_lookback, last_change,
        abs_sum_of_changes, abs_changes_lookback,
        get_baseline, baseline_shift, baseline_roll,
        sig_columns, sig_order, sig_lookback, sig_penoff, sig_addtime,
        irr_sig_columns, irr_sig_order, irr_sig_lookback, irr_sig_penoff, irr_sig_addtime):
    # Give the run object the save directory in case we want to save there
    _run.save_dir = save_dir + '/' + str(_run._id)

    # Get the data
    df, labels = get_eventual_sepsis_utilities(return_df=True)

    # Drop columns
    if drop_cols is not False:
        df.drop(drop_cols, axis=1, inplace=True)

    # Add absolute sum of changes
    if abs_sum_of_changes is not False:
        abs_changes_frame = AbsoluteSumOfChanges(lookback=abs_changes_lookback).transform(df)

    # Add moments
    if moments is not False:
        irr = AddMoments(moments=moments, lookback=moment_irr_lookback).transform(df[irregular_cols], return_full=False)
        cts = AddMoments(moments=moments, lookback=moment_lookback).transform(df[cts_cols], return_full=False)
        moment_frame = pd.concat([irr, cts], axis=1)

    # Add a the difference from a rolling mean window - 10
    if get_baseline is not False:
        rolling_mean = df.groupby('id').apply(lambda x: x.shift(baseline_shift).rolling(baseline_roll, min_periods=2).mean())
        baseline_diff = df - rolling_mean
        baseline_diff.columns = [x + '_basediff' for x in df.columns]

    # Add most recent changes
    if last_change is not False:
        change_frame = MostRecentChange(columns=last_change).transform(df, return_full=False)

    # Compile together
    df = pd.concat([
        df,
        baseline_diff if 'baseline_diff' in locals() else None,
        moment_frame if 'moment_frame' in locals() else None,
        abs_changes_frame if 'abs_changes_frame' in locals() else None,
    ], axis=1)

    # Add signatures
    if sig_columns is not False:
        df = add_signatures(
            df, sig_columns, order=sig_order, lookback=sig_lookback, pen_off=sig_penoff, addtime=sig_addtime, individual=True
        )

    # Add irregular signatures
    if irr_sig_columns is not False:
        df = add_irregular_signatures(
            df, irr_sig_columns, irr_sig_lookback, 3, True, True, irr_sig_addtime, irr_sig_penoff
        )

    # Now run a cv predict
    cv_iter = list(CustomStratifiedGroupKFold(n_splits=5).split(df, labels, groups=df.index.get_level_values('id')))
    predictions = cross_val_predict_to_series(XGBRegressor(), df, labels.values, cv=cv_iter)

    # AUC and save
    auc = roc_auc_score((labels >= 0).astype(int), predictions)
    ppprint('AUC: {:.3f}'.format(auc), color='green')
    save_pickle(predictions, _run.save_dir + '/probas.pickle')


if __name__ == '__main__':
    # Create an experiment to predict sepsis vs non sepsis
    experiment_name = input('Experiment name: ')
    # experiment_name = 'testing'
    ppprint("Running experiment '{}'".format(experiment_name), start=True)

    # Set save dir
    save_dir = create_fso(ex, experiment_name, dir=MODELS_DIR + '/sepsis_only')

    # Define the parameter grid
    param_grid = {
        'save_dir': [save_dir],
        'drop_cols': [
            ['hospital'],
            # *[[col] for col in drop_cols]
        ],

        # Moments
        'moments': [2],
        'moment_lookback': [7],
        'moment_irr_lookback': [7],

        # Other features
        'abs_sum_of_changes': [True],
        'abs_changes_lookback': [7, 10, 15],

        # Baseline
        'get_baseline': [False],
        'baseline_shift': [7],
        'baseline_roll': [10],

        # Signatures
        'sig_columns': [
            # False,
            # ['FiO2']
            ['SOFA', 'MAP']
            # ['BUN/CR']
            # ['Temp']
        ],
        'sig_lookback': [7],
        'sig_order': [3],
        'sig_penoff': [False],
        'sig_addtime': [False],

        'irr_sig_columns': [
            # False,
            ['FiO2']
            # ['FiO2'], ['Glucose'], ['SaO2']
            # ['Temp']
        ],
        'irr_sig_lookback': [7],
        'irr_sig_order': [3],
        'irr_sig_penoff': [False],
        'irr_sig_addtime': [False]
    }

    # Make the same every time
    for i, params in enumerate(ParameterGrid(param_grid)):
        ppprint('Run number ' + str(i + 1) + '\n' + '-' * 50, color='magenta')
        pprint(params)
        ex.run(config_updates=params)
