from definitions import *
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from src.features.transformers import SignaturesFromIdDataframe


class BaseIDTransformer(TransformerMixin, BaseEstimator):
    """
    Base class when performing transformations over ids. One must implement a transform_id method.
    """
    def __init__(self):
        pass

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    @timeit
    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            df_transformed = df.groupby(['id'], as_index=False).apply(self.transform_id)
        elif isinstance(df, pd.Series):
            df_transformed = df.groupby(['id']).apply(self.transform_id)

        # Sometimes creates a None level
        if None in df_transformed.index.names:
            df_transformed.index = df_transformed.index.droplevel(None)

        return df_transformed


