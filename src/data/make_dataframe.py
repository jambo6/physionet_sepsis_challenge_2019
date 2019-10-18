from definitions import *
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from src.data.transformers import *


if __name__ == '__main__':
    # Main dataframe generator
    locations = [DATA_DIR + '/raw/' + x for x in ['training_A', 'training_B']]

    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')

    # Run full pipe
    data_pipeline = Pipeline([
        # ('create_dataframe', CreateDataframe(save=True)),
        ('input_count', AddRecordingCount()),
        ('imputation', CarryForwardImputation()),
        ('derive_features', DerivedFeatures()),
        # ('fill_missing', FillMean()),
        # ('drop_features', DropFeatures()),
        # ('min_maxxer', MinMaxSmoother()),
    ])
    df = data_pipeline.fit_transform(df)

    # Save
    save_pickle(df, DATA_DIR + '/interim/munged/df.pickle')
    save_pickle(data_pipeline, MODELS_DIR + '/pipelines/data_pipeline.dill', use_dill=True)

    # Labels -> scores
    labels = load_pickle(DATA_DIR + '/processed/labels/original.pickle')
    scores = LabelsToScores().transform(labels)
    save_pickle(scores['utility'], DATA_DIR + '/processed/labels/utility_scores.pickle')
    save_pickle(scores, DATA_DIR + '/processed/labels/full_scores.pickle')

    # Save the ids of those who eventually develop sepsis
    if_sepsis = labels.groupby('id').apply(lambda x: x.sum()) > 1
    ids = if_sepsis[if_sepsis].index
    save_pickle(ids, DATA_DIR + '/processed/labels/ids_eventual.pickle')

    idx = df[df['SOFA_deterioration'] == 1].index
    labels.loc[idx].sum() / labels.loc[idx].shape[0]
    idx = df[df['SOFA_deterioration'] != 1].index
    labels.loc[idx].sum() / labels.loc[idx].shape[0]
