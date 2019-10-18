from definitions import *
import copy
# Esig cannot be loaded in jupyter for some reason
try:
    import esig.tosig as ts
except:
    pass
import iisignature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.features.signatures.functions import *


class SignatureTransformer(BaseEstimator, TransformerMixin):
    """
    Main class for building a multi-step signature computation pipeline.

    Note: Input is usually a 3D array of shape nxmxr where n is the number of paths, m is the number of timepoints and r
    is the number of features. However the decorator 'signatures_from_dataframes' is used around methods for which this
    shape is necessary and has the effect of transforming a single m x r path (in the form of a dataframe) onto a 1xmxr
    path so the method should still work, though can on occasion break for odd reasons.

    Example usage:
        transformer = SignatureTransformer(
            order=2, logsig=True, add_time=False, leadlag=True
        )
        X_signatures = transformer.fit_transform(X_data)
    """
    def __init__(self, order=2, logsig=True, add_time=False, pen_off=False, cumsum=False,
                 leadlag=False, append_zero=False, dropna=True, remove_redundancies=True, column_names=None,
                 scaling=False, fname=False, use_esig=True):
        """
        :param order: (int) The order of the signature
        :param logsig: (bool) Set true for log signature
        :param add_time: (bool) Add time column
        :param pen_off: (bool) Implement pen off
        :param cumsum: (bool) Perform a cumulative sum of the feautures
        :param leadlag: (bool) Apply lead lag transform
        :param append_zero: (bool) Add [0, 0, 0, ..., 0] to the start of the path
        :param dropna: (bool) Remove any rows with a nan value
        :param remove_redundancies: (bool) Get rid of redundant features before returning
        :param column_names: (list) Listed names of features that correspond to input path,
        :param scaling: (list) Scaling values for path
        :param fname: (str) Set a filename to save if wanting to save.
        :param use_esig: (bool) Set true to use esig rather than iisig
        """
        self.order = order
        self.logsig = logsig
        self.add_time = add_time
        self.pen_off = pen_off
        self.cumsum = cumsum
        self.leadlag = leadlag
        self.append_zero = append_zero
        self.scaling = scaling
        self.dropna = dropna
        self.remove_redundancies = remove_redundancies
        self.column_names = column_names
        self.fname = fname
        self.use_esig = use_esig

    def fit(self, X=None, y=None):
        """ Build the pipeline """
        # First do modifications of the path, cumsum, shifting, zero append etc, then do time and leadlag
        pre_computation_steps = [
            ('append_zero', AppendZero()),
            ('shift_to_zero', ShiftToZero()),
            ('cumsum', CumulativeSum()),
            ('add_time', AddTime()),
            ('pen_off', PenOff()),
            ('leadlag', LeadLag()),
            ('scaling', ScaleColumns(self.scaling)),
            ('dropna', DropNa())
        ]

        # Get the selected steps
        true_vals = [key for (key, value) in self.get_params().items() if value is not False]
        pre_computation_steps = [x for x in pre_computation_steps if x[0] in true_vals]

        # Create the signature computation part
        computation_step = [('signatures', ComputeSignature(order=self.order, logsig=self.logsig, use_esig=self.use_esig))]

        # Get all steps and generate pipeline
        steps = pre_computation_steps + computation_step
        self.pipeline = Pipeline(steps)

        # Fit the pipeline
        self.pipeline.fit(X, y)

        return self

    @signatures_from_dataframe
    def transform(self, X):
        # Calculate the signatures
        X_transformed = self.pipeline.transform(X)
        return X_transformed


class ComputeSignature(BaseEstimator, TransformerMixin):
    def __init__(self, order=2, logsig=False, n_prev=False, use_esig=False):
        self.order = order
        self.logsig = logsig
        self.n_prev = n_prev
        self.use_esig = use_esig

    def fit(self, X, y=None):
        # This function takes a long time, do once instead of multiple times
        if not self.use_esig:
            if self.logsig:
                self.logsig_prepare = iisignature.prepare(X[0].shape[-1], self.order, 'DH')
        return self

    def transform_instance(self, path, y=None):
        # Get either logsig function or normal sig
        if not self.logsig:
            if self.use_esig:
                signature = ts.stream2sig(path, self.order)
            else:
                signature = iisignature.sig(path, self.order)
        else:
            if self.use_esig:
                signature = ts.stream2logsig(path, self.order)
            else:
                signature = iisignature.logsig(path, self.logsig_prepare)
        return signature

    @single_path_method
    def transform(self, X):
        # Compute signatures
        X_signatures = np.array([self.transform_instance(path) for path in X])
        return X_signatures


class PenOff(BaseEstimator, TransformerMixin):
    """
    Performs the pen on pen off method to every path in an array of paths
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        # Add dimension of ones
        X_transformed = np.c_[X, np.ones(len(X))]

        # Pen down to 0, update
        last = np.array(copy.deepcopy(X_transformed[-1]))
        last[-1] = 0.
        X_transformed = np.r_[X_transformed, [last]]

        # Add home
        home = np.zeros(X_transformed.shape[1]).reshape(1, -1)
        X_transformed = np.r_[X_transformed, home]

        return X_transformed

    @single_path_method
    def transform(self, X, y=None):
        return np.array([self.transform_instance(x) for x in X])


class AddTime(BaseEstimator, TransformerMixin):
    """
    Adds a time dimension to every path in an array of paths.

    :param scales: (t_scale, feature_scale), will scale the paths by the specified number
    """
    def __init__(self, scales=[1, 1]):
        self.t_scale, self.feature_scale = scales

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        time = np.arange(X.shape[0]).reshape(-1, 1)
        # time_scaled = self.feature_scale * time
        # X_scaled = self.feature_scale * X
        # if len(X_scaled.shape) is 1:
        #     X_scaled = X_scaled.reshape(-1, 1)
        X_with_time = np.concatenate([time, X], axis=1)
        return X_with_time

    @single_path_method
    def transform(self, X):
        X_with_time = np.array([self.transform_instance(x) for x in X])
        return X_with_time


class RemoveNanRows(BaseEstimator, TransformerMixin):
    """
    Removes rows that contain a nan value. If, after removing nans, no values exist, then a nan array according to the
    shape of the path is inserted, so the signatures will be all zero
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        path = X[~np.isnan(X).any(axis=1)]
        # If nan remove removed all the entries, fill with a nan val
        if path.shape[0] == 0:
            path = np.full((1, path.shape[-1]), np.nan)
        return path

    def transform(self, X):
        X = np.array([self.transform_instance(x) for x in X])
        return X


class LeadLag(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        # Duplicate the rows of X, turn into array of shape of dimension (2*X.shape[0], X.shape[1])
        X_duplicated = np.array([[X[i, :], X[i, :]] for i in range(X.shape[0])]).reshape(2 * X.shape[0], X.shape[1])

        # Lead is all but the first, lag all but the last
        lead = X_duplicated[1:]
        lag = X_duplicated[:-1]
        leadlag = np.concatenate([lead, lag], axis=1)

        return leadlag

    @single_path_method
    def transform(self, X):
        return np.array([self.transform_instance(x) for x in X])


class CumulativeSum():
    def __init__(self, append_zero=True):
        self.append_zero = append_zero


    def fit(self, X, y=None):
        return self

    def transform_instance(self, path):
        # Make cumsum path
        path_cs = np.cumsum(path, axis=0)

        # Add zero to get proper moments
        if self.append_zero:
            zeros = np.zeros(shape=(1, path.shape[-1]))
            path_cs = np.concatenate([zeros, path_cs], axis=0)

        return path_cs

    @single_path_method
    def transform(self, X):
        X_cumsum = [self.transform_instance(path) for path in X]
        return X_cumsum


class RemoveRedundantFeatures(BaseEstimator, TransformerMixin):
    """
    If any columns have only a single value, then the feature is redundant and can be removed. This happens for
    example in time signatures, any time only columns are unimportant and can be removed
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Make into a dataframe, and drop columns that have a single unique value
        df = pd.DataFrame(X.T)
        nunique = df.apply(pd.Series.nunique, axis=1)
        rows_to_drop = list(nunique[nunique == 1].index)

        # Remove duplicated columns
        rows_to_drop.extend(list(df[df.duplicated()].index))

        # Drop em
        df.drop(rows_to_drop, inplace=True)

        return df.values.T, list(rows_to_drop)


class AppendZero():
    """ This will append a zero starting vector to every path """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @single_path_method
    def transform(self, X):
        zero_vec = np.zeros(shape=(1, X[0].shape[-1]))
        return np.array([np.concatenate([zero_vec, path], axis=0) for path in X])


class ShiftToZero():
    """ This will shift every path to begin at the origin """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @single_path_method
    def transform(self, X):
        return np.array([path - path[0, :] for path in X])


def get_new_column_names(order, num_features, drop_idxs=None):
    """
    Outputs information about what variables the signature has come from.

    Example: We calculate the signatures for a dataset with 2 features, suppose self.order=2. Then put num_features
    = 2 and the following list will be returned
        signature_names = ['1|(1)', '1|(2)', '2|(1, 1)', '2|(1, 2)', '2|(2, 1)', '2|(2, 2)']
    the first number specifies the order, the second the combination of features used in its creation

    :param num_features:
    :param drop_idxs:
    :return:
    """
    # Get the signature ids in a list
    sig_keys = ts.sigkeys(num_features, order)
    sig_keys = listed_sig_keys(sig_keys)

    # Reindex to start at 0
    sig_keys_reindexed = [list(np.array(x)-1) for x in sig_keys]

    # Signature column names
    sig_col_names = [str(len(x)) + '|' + str(x).replace('[', '(').replace(']', ')') for x in sig_keys_reindexed]

    # Drop some if specified
    if drop_idxs is not None:
        sig_col_names = [sig_col_names[i] for i in range(len(sig_col_names)) if i not in drop_idxs]

    return sig_col_names, sig_keys_reindexed


def listed_sig_keys(ndim=None, order=None, logsig=False):
    """ Parse the sig_key sting into a more useful list """
    # Calculate signature keys dependent on whether we logsignatured or not
    sig_keys = ts.logsigkeys(ndim, order) if logsig else ts.sigkeys(ndim, order)

    # Split out the string
    if logsig:
        sig_key_list = sig_keys.split(' ')[1:]
        sig_key_list = [x if isinstance(x, list) else [x] for x in list(map(eval, sig_key_list))]
    else:
        sig_key_list = sig_keys.replace('(', '[').replace(')', ']').split(' ')
        sig_key_list = list(map(eval, sig_key_list[1:]))

    # Convert to tuples to make for nicer reading
    sig_key_list = [tuple(x) for x in sig_key_list]

    return sig_key_list


class ScaleColumns(BaseEstimator, TransformerMixin):
    """ Scales columns by given amount """
    def __init__(self, scaling=False):
        """
        :param scaling: (list) example [1, 2, 0.5] will scales column 1 by 1, 2 by 2 and 3 by 0.5
        """
        self.scaling = scaling

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert X[0].shape[-1] == len(self.scaling)
        return [x * self.scaling for x in X]


class DropNa(TransformerMixin, BaseEstimator):
    """
    Given a list of paths, this will loop through and remove any rows that contain a nan value. If we end up with an
    empty array, it fills this with all zeros and the correct shape.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fill any empty path with a zero path
        zero_path = np.zeros(shape=X[0].shape)

        # Remove any row with a nan
        X = [x[~np.isnan(x).any(axis=1)] for x in X]

        # Fill with the zero path if all entries were nan
        X = [x if x.shape[0] != 0 else zero_path for x in X]

        return X


# Some testing
if __name__ == '__main__':
    X = np.random.randn(3, 1)
    a = PenOff().transform([X])
    sig = ComputeSignature(order=3).transform(a)
