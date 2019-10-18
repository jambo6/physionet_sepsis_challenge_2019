"""
Rebuilds and trains the time subsetted model
"""
from sklearn.pipeline import Pipeline
from src.features.transformers import SignaturesFromIdDataframe, IrrSignaturesFromIdDataframe
from src.data.transformers import *
from submission.generate_submission.build_main import TrainModel


def make_signature_pipeline(config):
    options = {
        'order': 2,
        'leadlag': True,
        'logsig': True,
        'use_esig': True,
    }

    config['columns'] = ['SOFA', 'MAP', 'Temp']

    sig_tuples = [('signatures_{}'.format(column),
                   SignaturesFromIdDataframe(columns=[column], lookback=6, options=options))
                  for column in config['columns']]

    signature_pipeline = Pipeline(sig_tuples)
    return signature_pipeline


def make_data_pipeline(config):
    data_pipeline = Pipeline([
        # ('create_dataframe', CreateDataframe(save=True)),
        ('imputation', CarryForwardImputation()),
        ('derive_features', DerivedFeatures()),
        ('drop_features', DropFeatures(features=['hospital']))
    ])
    return data_pipeline


if __name__ == '__main__':
    # Save file
    save_location = MODELS_DIR + '/solutions/time_subset/solution_1.pickle'

    # Load raw frames
    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    labels_binary = load_pickle(DATA_DIR + '/processed/labels/original.pickle')
    labels_utility = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')

    # Load best configuration file
    model_dir = ROOT_DIR + '/models/experiments/time_subset/new_mean_std_sigs/1'
    # config = load_json(model_dir + '/config.json')
    config={}

    # Reduce the df
    df = df.loc[pd.IndexSlice[:, :58], :]
    labels_binary, labels_utility = labels_binary.loc[df.index], labels_utility.loc[df.index]

    # Get data pipe
    data_pipeline = make_data_pipeline(config)

    # Add means and stds
    other_feature_pipeline = Pipeline([
        ('std', GetChunkStatistic(stat='std', lookback=6))
    ])

    # Add signatures
    signature_pipeline = make_signature_pipeline(config)

    # Make full pipe
    pipeline = Pipeline([
        ('setup_data', data_pipeline),
        ('other_features', other_feature_pipeline),
        ('compute_signatures', signature_pipeline),
        ('train_model', TrainModel())
    ])

    pipeline.fit(df, labels_utility)
    predictions = pipeline.predict(df)
    print(pipeline.score(df, labels_binary))

    save_pickle(pipeline, save_location)
