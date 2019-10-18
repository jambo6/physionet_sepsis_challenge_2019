""" Experiment to determine any new features from tsfresh """
# Set experiment
from sacred import Experiment
ex = Experiment('main_experiment')

# Imports
from definitions import *
from pprint import pprint
import scipy.stats as stats
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from xgboost import XGBClassifier
from src.models.functions import *
from src.models.susceptible.functions import *
from src.models.experiments.functions import *
from src.data.transformers import DerivedFeatures


@ex.config
def config():
    save_dir = None
    detection = 'susceptible'
    moments = False
    tsfresh = None

@ex.main
def run(_run, save_dir, detection, moments, tsfresh):
    # Add id to save dir
    _run.save_dir = save_dir + '/' + str(_run._id)

    # Get the data
    data_raw, labels_binary_raw, _ = load_munged_data()
    labels_eventual = make_eventual_sepsis_labels(labels_binary_raw)

    # Get the first n
    if detection == 'susceptible':
        data, labels = first_n_timepoints(data_raw, labels_eventual, n=30)
    elif detection == 'sepsis':
        data, labels = get_long_time_sepsis_cases(data_raw, labels_eventual)

    # Compute some basic features
    features = data.groupby('id').apply(lambda x: x.mean())
    features.columns = [x + '_mean' for x in features.columns]

    if moments is not False:
        moments = add_statistical_moments(data, up_to=moments)
        features = pd.concat([features, moments], axis=1)

    # Compute tsfresh features
    if tsfresh is not False:
        # Make tsfresh form
        print(tsfresh)
        data_tsfresh, labels_single = to_tsfresh_form(data[tsfresh], labels)

        # Impute, else tsfresh fails
        data_imputed = impute(data_tsfresh)

        # Extract best features using tsfresh algo
        tsfresh_features = extract_relevant_features(data_imputed, labels_single,
                                                     column_id='id', column_sort='time', n_jobs=5)

        features = pd.concat([features, tsfresh_features], axis=1)

        # Save the features
        _run.log_scalar('num_tsfresh_features', len(features.columns))
        save_pickle(tsfresh_features.columns, _run.save_dir + '/tsfresh_features.pickle')
    else:
        labels_single = labels.groupby('id').apply(lambda x: x.iloc[0])

    # Now run a cv predict
    cv_iter = list(StratifiedKFold(n_splits=5, random_state=2).split(features, labels_single))
    probas = cross_val_predict(XGBClassifier(), features, labels_single, cv=cv_iter, n_jobs=5, method='predict_proba')[:, 1]
    # probas = cross_val_predict(XGBClassifier(), features, labels_single, cv=cv_iter, n_jobs=5)

    # Save some metrics
    auc = roc_auc_score(labels_single, probas)
    # auc = accuracy_score(labels_single, probas)
    _run.log_scalar('auc', auc)

    # Save probas
    proba_frame = pd.Series(index=features.index, data=probas)
    save_pickle(proba_frame, _run.save_dir + '/probas.pickle')

    # Print results
    ppprint('AUC: {:.3f}'.format(auc), color='green')



if __name__ == '__main__':
    # Some model setup
    experiment_name = input('Experiment name: ')
    ppprint("Running experiment '{}'".format(experiment_name), start=True)

    # Set save dir
    save_dir = create_fso(ex, experiment_name, dir=MODELS_DIR + '/susceptible/tsfresh')

    # Define the parameter grid
    param_grid = {
        'save_dir': [save_dir],
        # 'detection': ['susceptible', 'sepsis'], # Detect the differences in the susceptible group, or the sepsis differences
        'detection': ['susceptible'],
        'moments': [2, 3, 4],
        'tsfresh': [
            False,
            # ['SBP', 'Temp', 'Resp', 'DBP']
            # ['ShockIndex'], ['MAP'], ['Temp'], ['O2Sat'], ['SBP'], ['DBP'], ['Resp'],
            # ['FiO2'], ['Lactate'], ['SaO2'], ['Platelets']
        ]
    }

    # Make the same every time
    for i, params in enumerate(ParameterGrid(param_grid)):
        ppprint('Run number ' + str(i + 1) + '\n' + '-' * 50, color='magenta')
        pprint(params)
        ex.run(config_updates=params)







