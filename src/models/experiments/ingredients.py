from sacred import Ingredient
data_ingredient = Ingredient('data')
feature_ingredient = Ingredient('feature')
train_ingredient = Ingredient('train')

# My imports
from lightgbm import LGBMRegressor, LGBMClassifier
from src.models.experiments.functions import *
from src.features.transformers import *
from src.models.functions import *
from src.features.feature_selection import FeatureSelector
from src.data.extracts import irregular_cols, cts_cols
from src.models.optimizers import old_params, ThresholdOptimizer
from sklearn.metrics import roc_auc_score


# Feature Generation
@feature_ingredient.config
def feature_config():
    # Use RFECV feature selection
    feature_selction = False

    # Num measurements in the last n_hours
    num_measurements = False

    # Moments
    moments = False
    moment_lookback = 6

    # Signature options
    columns = False
    lookback = False
    lookback_method = 'mean'
    individual = False
    order, logsig, leadlag, addtime, cumsum, pen_off, append_zero = 2, True, False, False, False, False, False

    # Cumsum signatures
    cs_columns = False
    cs_lookback = 10
    cs_order = 3
    cs_logsig = True
    cs_leadlag = True
    cs_addtime = False

    # Other
    extra_features = False
    add_max, add_min = False, False
    max_min_lookback = 5
    drop_count = False

    # EXTRA
    mean_reduced_cols = False
    mean_reduced_sig_cols = False
    mean_reduced_moment_cols = False
    drop_count_moments = False
    drop_specific = False
    drop_specific_all = False
    extra_sofa_sigs = False
    extra_temp = False
    irr_max_min_lb = False

    # For submission
    last_only = False

@feature_ingredient.capture
def generate_features(_run,
    num_measurements, moments, moment_lookback,
    columns, lookback, lookback_method, individual, order, logsig, leadlag, addtime, cumsum, pen_off, append_zero,
    cs_columns, cs_lookback, cs_order, cs_logsig, cs_leadlag, cs_addtime,
    add_max, add_min, max_min_lookback,
    last_only):
    # Get data
    df, labels_binary, labels_utility = load_munged_data()
    labels_eventual = load_pickle(DATA_DIR + '/processed/labels/eventual_sepsis.pickle')
    df.drop('hospital', axis=1, inplace=True)

    # Get number of measurements taken in a fixed time window
    counts_24hrs = None
    if num_measurements is not False:
        cols = [x for x in df.columns if '_count' in x]
        counts_24hrs = GetNumMeasurements(lookback=num_measurements).transform(df[cols])
        counts_24hrs = pd.DataFrame(index=df.index, data=counts_24hrs, columns=['{}_cntxhrs'.format(x) for x in cols])

    # Moments
    moments_frame = None
    if moments is not False:
        moments_frame = AddMoments(moments=moments, lookback=moment_lookback, last_only=last_only).transform(df)
        moments_frame = numpy_to_named_dataframe(moments_frame, df.index, 'Moments')
        moments_frame.columns = ['{}_moment_{}'.format(col, i) for i in range(2, moments + 1) for col in df.columns]

    # Add signatures
    signatures = None
    if columns is not False:
        signatures = add_signatures(df, columns, individual, lookback, lookback_method,
                                    order, logsig, leadlag, addtime, cumsum, pen_off, append_zero, last_only=last_only)
        signatures = numpy_to_named_dataframe(signatures, df.index, 'Signatures')

    # Add cumsum signatures
    cs_signatures = None
    if cs_columns is not False:
        cs_signatures = add_signatures(df, cs_columns, individual, cs_lookback, lookback_method,
                                    cs_order, cs_logsig, cs_leadlag, cs_addtime, True, pen_off, append_zero, last_only=last_only)
        cs_signatures = numpy_to_named_dataframe(cs_signatures, df.index, 'CsSignatures')

    # Get sampling rate rather than absolute number for the count column.
    data = GetRateOfLaboratorySampling().transform(df)

    # Max and min
    cols = cts_cols
    max_vals = None
    if add_max is not False:
        max_vals = GetStatistic(statistic='max', lookback=max_min_lookback, columns=cols).transform(df[cols])
        max_vals = pd.DataFrame(index=df.index, data=max_vals, columns=['{}_max'.format(x) for x in cols])

    min_vals = None
    if add_min is not False:
        min_vals = GetStatistic(statistic='min', lookback=max_min_lookback, columns=cols).transform(df[cols])
        min_vals = pd.DataFrame(index=df.index, data=min_vals, columns=['{}_min'.format(x) for x in cols])

    # Create data ready for insertion
    df = pd.concat([data, counts_24hrs, moments_frame, signatures, max_vals, min_vals, cs_signatures], axis=1)
    # data = np.concatenate([x for x in (data, counts_24hrs, moments_frame, signatures) if x is not None], axis=1)

    # Add to run
    _run.df, _run.labels_binary, _run.labels_utility, _run.labels_eventual = df, labels_binary, labels_utility, labels_eventual

    return df

# TRAIN
@train_ingredient.config
def train_config():
    cv_hospital = False
    gs_params = False
    n_estimators = 100
    learning_rate = 0.1
    binary = False

@train_ingredient.capture
def train_model(_run, cv_hospital, gs_params, n_estimators, learning_rate, binary):
    # Load
    df, labels_binary, labels_utility, labels_eventual = _run.df, _run.labels_binary, _run.labels_utility, _run.labels_eventual

    # Get the cross validated folds
    if cv_hospital:
        df_hosp, _, _ = load_munged_data()
        hosp = df_hosp['hospital'].values
        t1, t2 = np.argwhere(hosp == 1).reshape(-1), np.argwhere(hosp == 2).reshape(-1)
        cv_iter = [(t1, t2), (t2, t1)]
        # cv_iter = CustomStratifiedGroupKFold(n_splits=2).split(df, labels_binary, groups=df.index.get_level_values('id'))
    else:
        cv_iter = CustomStratifiedGroupKFold(n_splits=5).split(df, labels_binary, groups=df.index.get_level_values('id'))

    if binary == False:
        # Setup the classifier
        if gs_params is not False:
            params = load_pickle(MODELS_DIR + '/parameters/lgb/random_grid_fullds.pickle')
            clf = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate).set_params(**params).set_params(**{'min_child_samples': 199, 'min_child_weight': 38, 'num_leaves': 49})
        else:
            clf = LGBMRegressor(**old_params).set_params(**{'n_estimators': n_estimators, 'learning_rate': learning_rate})

        # Make predictions
        predictions = cross_val_predict_to_series(clf, df, labels_utility, cv=cv_iter, n_jobs=-1)

        # Perform thresholding
        binary_preds, scores, _ = ThresholdOptimizer(budget=100, labels=labels_binary, preds=predictions).cross_val_threshold(cv_iter, parallel=True, give_cv_num=True)

        # Log results
        ppprint('\tAVERAGE SCORE {:.3f}'.format(np.mean(scores)), color='green')
        _run.log_scalar('utility_score', np.mean(scores))
        save_pickle(predictions, _run.save_dir + '/probas.pickle')

    elif binary == True:
        # Set classifier
        if gs_params:
            params = load_pickle(MODELS_DIR + '/parameters/lgb/random_grid_fullds.pickle')
            clf = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate).set_params(**params)
            print(params)
            print(clf)
        else:
            print(df.shape)
            clf = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

        # Make full predictions
        predictions = cross_val_predict_to_series(clf, df, labels_utility, cv=cv_iter, n_jobs=-1, method='predict_proba')

        # Reduce to eventual
        labels, preds = labels_eventual.groupby('id').apply(max), predictions.groupby('id').apply(max)

        # Log results
        print(roc_auc_score(labels, preds))
        save_pickle(labels_eventual, MODELS_DIR + '/temp/eventual_preds/labels.pickle')
        save_pickle(predictions, MODELS_DIR + '/temp/eventual_preds/preds.pickle')
        save_pickle(labels, MODELS_DIR + '/temp/eventual_preds/labels_max.pickle')
        save_pickle(preds, MODELS_DIR + '/temp/eventual_preds/preds_max.pickle')

