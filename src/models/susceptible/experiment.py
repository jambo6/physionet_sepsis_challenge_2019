""" Experiment for predicting susceptibility """
from sacred import Experiment
ex = Experiment('susceptibility_experiment')

# My imports
from definitions import *
from pprint import pprint
from sklearn.model_selection import ParameterGrid, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from boruta import BorutaPy
from src.features.transformers import AddMoments, AddBaseline
from src.models.functions import *
from src.models.susceptible.functions import *
from src.models.experiments.functions import create_fso, print_experiment_params


# Setup experiment
@ex.config
def config():
    drop_positive_utility = True
    drop_iculos = False
    baseline = False
    moments = False
    sig_columns = False

    # Irr sigs
    irr_sig_columns = False
    irr_sig_lookback = 7
    irr_sig_addtime = True
    irr_sig_penoff = False

    # Feature selection
    feature_selection = False


@ex.main
def run(_run, save_dir, drop_positive_utility, drop_iculos, baseline, moments,
        sig_columns,
        irr_sig_columns, irr_sig_lookback, irr_sig_addtime, irr_sig_penoff,
        ):
    # Give the run object the save directory in case we want to save there
    _run.save_dir = save_dir + '/' + str(_run._id)

    # Get the data
    df, _, _ = load_munged_data()
    labels = get_eventual_sepsis_labels(drop_positive_utility=drop_positive_utility).astype(int)
    # df = df.loc[labels.index]

    if drop_iculos:
        df.drop('ICULOS', axis=1, inplace=True)

    # Reduce to < 58 time
    # df = df.loc[pd.IndexSlice[:, :58], :]
    # labels = labels.loc[pd.IndexSlice[:, :58]]

    # Add a the difference from a rolling mean window - 10
    if baseline is not False:
        baseline_diff = AddBaseline().transform(df)

    # Add means and stds
    moments_frame = None
    if moments is not False:
        moments_frame = AddMoments(moments=moments, lookback=7, over_paths=True).transform(df)
        # moments_frame = pd.DataFrame(index=df.index, data=moments_frame, columns=cols)

    # Add signatures
    signatures = None
    if sig_columns is not False:
        signatures = add_signatures(df, sig_columns, individual=True)

    # Add irregular signatures
    irr_signatures = None
    if irr_sig_columns is not False:
        irr_signatures = add_irregular_signatures(
            df, irr_sig_columns, irr_sig_lookback, 3, True, True, irr_sig_addtime, irr_sig_penoff
        )

    # Compile features
    data = np.concatenate([x for x in (df.values, moments_frame, signatures) if x is not None], axis=1)
    data = pd.DataFrame(index=df.index, data=data)

    # Now cv it
    clf = LGBMClassifier(n_estimators=100, max_depth=4, n_jobs=-1)
    cv_iter = list(CustomStratifiedGroupKFold(n_splits=5).split(data, labels, groups=df.index.get_level_values('id')))
    probas = cross_val_predict_to_series(clf, data, labels, cv=cv_iter, n_jobs=5, method='predict_proba')

    # AUC and save
    auc = roc_auc_score(labels, probas)
    ppprint('AUC: {:.3f}'.format(auc), color='green')
    _run.log_scalar(auc, 'auc')
    save_pickle(probas, _run.save_dir + '/probas.pickle')


if __name__ == '__main__':
    # Create an experiment to predict sepsis vs non sepsis
    experiment_name = input('Experiment name: ')
    # experiment_name = 'testing'
    ppprint("Running experiment '{}'".format(experiment_name), start=True)

    # Set save dir
    save_dir = create_fso(ex, experiment_name, dir=MODELS_DIR + '/experiments/susceptibility')

    # Define the parameter grid
    param_grid = {
        'save_dir': [save_dir],
        'drop_positive_utility': [False],
        'drop_iculos': [False],
        'baseline': [False],

        # Moments
        'moments': [4],

        # Signatures
        'sig_columns': [
            False,
            # ['SOFA'],
            # ['SOFA', 'MAP', 'ShockIndex'],
        ],

        # Irr sigs
        'irr_sig_columns': [
            False,
            # ['FiO2', 'SaO2', 'Glucose']
            # ['FiO2', 'SaO2', 'WBC', 'Lactate', 'Platelets', 'BUN', 'Hct', 'Hgb', 'Resp', 'MAP'],
        ],
        'irr_sig_lookback': [30],
        'irr_sig_addtime': [True],
        'irr_sig_penoff': [False],
    }


    # Make the same every time
    for i, params in enumerate(ParameterGrid(param_grid)):
        ppprint('Run number ' + str(i + 1) + '\n' + '-' * 50, color='magenta')
        print_experiment_params(param_grid, params)
        ex.run(config_updates=params)
