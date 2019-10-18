from definitions import *
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from src.models.functions import load_munged_data, add_signatures
from src.data.transformers import CarryForwardImputation, DerivedFeatures, AddRecordingCount
from src.features.transformers import AddMoments, GetNumMeasurements, GetRateOfLaboratorySampling, GetStatistic
from src.models.optimizers import ThresholdOptimizer
from src.features.feature_selection import FeatureSelector
from src.models.functions import numpy_to_named_dataframe, remove_useless_columns
from src.data.extracts import cts_cols

# Choose the model run to use
# model_dir = MODELS_DIR + '/experiments/main/newsol/1'
model_dir = ROOT_DIR + '/models/experiments/main/new_solution_10/1'
config = load_json(model_dir + '/config.json')


def basic_data_process(df=None):
    if SUBMISSION:
        save_pickle(df.columns, MODELS_DIR + '/other/col_names.pickle')
        # Drop hospital col
        df.drop(['hospital'], axis=1, inplace=True, errors='ignore')

        # Setup pipe
        data_pipeline = Pipeline([
            ('input_count', AddRecordingCount(last_only=True)),
            ('imputation', CarryForwardImputation()),
            ('derive_features', DerivedFeatures()),
        ])

        # Transform
        df = data_pipeline.transform(df)
        df.drop('hospital', inplace=True, axis=1, errors='ignore')
    else:
        df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
        df.drop(['hospital'], axis=1, inplace=True, errors='ignore')

    return df


def wrapper(df, config, submission=False):
    val = 100009
    from copy import deepcopy
    df_reduced = deepcopy(df).loc[[val]]
    df_single, data_single = process_data_and_build(df_reduced, config['feature'], True)

    # Make new
    # SUBMISSION = False
    # global SUBMISSION
    df_reduced = deepcopy(df).loc[[val]]
    df_all, data_all = process_data_and_build(df_reduced, config['feature'], False)

    # Compare
    df_all_single = df_all.iloc[[-1]].loc[val]
    df_single = pd.DataFrame(index=df_all_single.index, data=data_single, columns=df_all_single.columns)
    df_all_single, df_single = df_all_single.loc[29], df_single.loc[29]

    # Check nan cols are the same
    single_na, all_single_na = df_single[df_single.isna()].index, df_all_single[df_all_single.isna()].index
    len([x for x in all_single_na if x not in single_na])
    df_all_single[df_all_single != df_single].shape

    pass


def process_data_and_build(df, config, submission=False):
    # Process
    idx = df.index if not submission else df.iloc[[-1]].index.unique()

    # df = df.loc[[9]]
    # Get number of measurements taken in a fixed time window
    counts_24hrs = None
    if config['num_measurements'] is not False:
        cols = [x for x in df.columns if '_count' in x]
        counts_24hrs = GetNumMeasurements(lookback=config['num_measurements'], last_only=submission).transform(df[cols])
        if submission is False:
            counts_24hrs = pd.DataFrame(index=df.index, data=counts_24hrs, columns=['{}_cntxhrs'.format(x) for x in cols])

    # Moments
    moments_frame = None
    if config['moments'] is not False:
        moments_frame = AddMoments(moments=config['moments'], lookback=config['moment_lookback'], last_only=submission).transform(df)
        if submission is False:
            moments_frame = numpy_to_named_dataframe(moments_frame, df.index, 'Moments')
            moments_frame.columns = ['{}_moment_{}'.format(col, i) for i in range(2, config['moments'] + 1) for col in df.columns]
        if config['drop_count_moments']:
            moments_frame.drop([x for x in moments_frame.columns if 'count_moment' in x], axis=1, inplace=True)

    # Add signatures
    signatures = None
    if config['columns'] is not False:
        signatures = add_signatures(df, config['columns'], config['individual'], config['lookback'], config['lookback_method'],
                                    config['order'], config['logsig'], config['leadlag'], config['addtime'], config['cumsum'], config['pen_off'], config['append_zero'],
                                    last_only=submission)
        if submission is False:
            signatures = numpy_to_named_dataframe(signatures, idx, 'Signatures')

    # Max and min
    cols = cts_cols
    max_vals = None
    if config['add_max'] is not False:
        max_vals = GetStatistic(statistic='max', lookback=config['max_min_lookback'], columns=cols).transform(df[cols])
        if submission is False:
            max_vals = pd.DataFrame(index=df.index, data=max_vals, columns=['{}_max'.format(x) for x in cols])

    min_vals = None
    if config['add_min'] is not False:
        min_vals = GetStatistic(statistic='min', lookback=config['max_min_lookback'], columns=cols).transform(df[cols])
        if submission is False:
            min_vals = pd.DataFrame(index=df.index, data=min_vals, columns=['{}_min'.format(x) for x in cols])

    # Get sampling rate rather than absolute number for the count column.
    data = GetRateOfLaboratorySampling(last_only=submission).transform(df)

    # Create data ready for insertion
    # df.drop([x for x in df.columns if '_count' in x], axis=1, inplace=True)
    df = pd.concat([data, counts_24hrs, moments_frame, signatures, max_vals, min_vals], axis=1)
    # data = np.concatenate([x for x in (data, counts_24hrs, moments_frame, signatures) if x is not None], axis=1)
    # df = remove_useless_columns(df)

    # Drop the count col
    if config['drop_count']:
        df.drop([x for x in df.columns if '_count' in x], axis=1, inplace=True)

    # Drop specific cols
    if config['drop_specific'] is not False:
        df.drop(config['drop_specific'], axis=1, inplace=True)

    # Drop all that have the name prefix
    if config['drop_specific_all'] is not False:
        for col in config['drop_specific_all']:
            df.drop([x for x in df.columns if col in x], axis=1, inplace=True)

    # Print info about the shape
    print(df.shape)
    print(df.columns)

    # Assert that all cntxhrs cols are not nan because kept happening in training
    cntxhrscols = [x for x in df if 'xhrs' in x]
    assert df[cntxhrscols].isna().sum().sum() != df[cntxhrscols].shape[0]*df[cntxhrscols].shape[1], 'THING IS HAPPENEING'

    save_pickle(df.loc[[9]], './original_9.pickle')
    print("DONE")

    return df


def train_classifier(df, labels_binary, labels_utility, config=None):
    # Train
    params = load_pickle(MODELS_DIR + '/parameters/lgb/random_grid_fullds.pickle')
    clf = LGBMRegressor(n_estimators=2000, learning_rate=0.006, n_jobs=-1).set_params(**params)
    clf.fit(df, labels_utility)
    # predictions_full = pd.Series(index=df.index, data=clf.predict(df.values))
    # print(predictions_full.shape)
    predictions_full = pd.Series(index=labels_utility.index, data=clf.predict(df))

    threshold, score = ThresholdOptimizer(budget=1000).optimize_utility(labels_binary, predictions_full)

    return clf, threshold, score, predictions_full


if __name__ == '__main__':
    # Load in the data
    # df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    _, labels_binary, labels_utility = load_munged_data()

    # Save the column names
    # save_pickle(df.columns, MODELS_DIR + '/other/col_names.pickle')

    # Sort data
    # df = basic_data_process(None)
    df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
    df.drop(['hospital'], axis=1, inplace=True, errors='ignore')
    # df = wrapper(df, config, submission=False)
    df = process_data_and_build(df, config['feature'], submission=False)

    # Train final algo
    clf_final, threshold, score, predictions_full = train_classifier(df, labels_binary, labels_utility, config=config['train'])

    # Save
    save_loc = MODELS_DIR + '/submissions/submission_9'
    print('SAVE')
    save_pickle(clf_final, save_loc + '/clf.pickle')
    save_pickle(predictions_full, save_loc + '/predictions.pickle')
    save_pickle(threshold, save_loc + '/threshold.pickle')

    # Print info
    ppprint('SCORE: {:.3f}'.format(score), color='green')

