from definitions import *
from xgboost import XGBRegressor
from src.models.optimizers import ThresholdOptimizer, apply_threshold
from src.models.evaluators import ComputeNormalizedUtility


class TrainModel(BaseEstimator, TransformerMixin):
    """ Allows for both model fitting and threshold optimization to be put in the pipeline """
    def __init__(self, threshold_budget=300, clf=XGBRegressor()):
        self.budget = threshold_budget  # Num iterations for the threshold optimiser
        self.clf = clf

    def fit(self, df, labels_utility):
        # Fit the classifier
        self.clf.fit(df, labels_utility)

        # Make predictions
        predictions = self.clf.predict(df)

        # Optimise the threshold
        labels_binary = load_pickle(DATA_DIR + '/processed/labels/original.pickle')
        labels_binary = labels_binary.loc[df.index]
        assert labels_binary.shape[0] == df.shape[0]
        threshold, score = ThresholdOptimizer(budget=self.budget).optimize_utility(labels_binary, predictions)

        # Save the threshold as is used in prediction
        self.threshold = threshold

        return self

    def predict(self, df):
        """ Make predictions and apply the generated threshold for the result """
        predictions = self.clf.predict(df)
        predictions = apply_threshold(predictions, threshold=self.threshold)
        return predictions

    def score(self, df, labels):
        predictions = self.predict(df)
        score = ComputeNormalizedUtility().score(labels, predictions)
        return score

