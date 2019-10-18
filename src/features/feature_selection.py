from definitions import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from lightgbm import LGBMRegressor
from src.models.functions import load_munged_data, CustomStratifiedGroupKFold

# Rid some pointless warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """ Performs feature selection using an rfecv class. """
    def __init__(self, verbose=1):
        self.verbose = verbose  # Set zero for no print out

    def fit(self, df, labels):
        """ Fits assuming we use the normal cv split. """
        # Evaluate on a random cv split
        cv_iter = CustomStratifiedGroupKFold(seed=np.random.randint(1000)).split(df, labels)

        # Setup the rfe object
        rfe = RFECV(estimator=LGBMRegressor(n_estimators=1), cv=cv_iter, verbose=3)

        # Cant deal with nans so fill
        df.fillna(-10000, inplace=True)
        df = df.replace(np.inf, 2000)
        df = df.replace(-np.inf, 2000)

        # Fit the rfe
        rfe.fit(df.values, labels.values)

        # Find the best features
        num_feautures = rfe.n_features_
        selected = df.columns[rfe.support_]
        ordered_ranks = sorted(list(zip(df.columns, rfe.ranking_)), key=lambda x: x[1])

        # Print info and save
        if self.verbose != 0:
            print('Selected {} of {} features'.format(num_feautures, df.shape[-1]))
            for name, rank in ordered_ranks:
                print('Rank {}: {}'.format(rank, name))

        # Save
        save_pickle(list(selected), MODELS_DIR + '/feature_selection/rfecv_test.pickle')
        self.features = selected

        return self

    def transform(self, data):
        return data[self.features]



if __name__ == '__main__':
    # Load data
    df, labels_binary, labels_utility = load_munged_data()

    fs = FeatureSelector(verbose=1).fit(df, labels_utility)





