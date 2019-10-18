from definitions import *
import warnings
from copy import deepcopy
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from src.features.signatures.transformers import SignatureTransformer, RemoveNanRows
from src.data.extracts import irregular_cols, cts_cols


class MakePaths():
    """ Creates paths for each id from each timepoint given a number of lookback times.

    Input a dataframe:
        1. Loops over ids
        2. Pads the dataframe with num_lookback times so we can get paths close to t=0
        3. Gets paths of variables from each timepoint of lookback length
        4. Appends paths to a list and returns

    TODO allow for a method that does not pad, and just uses less timepoints near 0 time
    """
    def __init__(self, lookback=6, method='max', last_only=False):
        """
        :param lookback: The length of the lookback window
        :param padding: Takes a number of options:
            - 'fixed' pads all paths to the same length, so 3rd path with lookback 5 will be [1st, 1st, 1st, 2nd, 3rd]
            - 'max' will use no padding, so the paths at the start have a reduced length
            - 'mean' is similar to max but once we pass the lookback length, the start point is taken to be the mean
            of the values up to that point
            - 'pad_zero' pads to equal length with the zero value
        :param last_only: Returns the last path only. This is needed for submission algos.
        """
        self.lookback = lookback
        self.method = method
        self.last_only = last_only

    @staticmethod
    def pad_start(data, num):
        """ Pads the data with the initial values appended num times """
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)
        padding = np.repeat(data[0], num).reshape(num, -1, order='F')
        padded_data = np.concatenate([padding, data])
        return padded_data

    @staticmethod
    def mean_start(data, lookback):
        path = deepcopy(data)
        if path.shape[0] > lookback:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(path[:path.shape[0] - lookback + 1], axis=0)
            path = path[path.shape[0]-lookback:]
            path[0] = mean
        return path

    def transform(self, df):
        # Get the ids, setup the array to store all the paths
        ids = df.index.get_level_values('id').unique()
        all_paths = []

        # For each id, find all lookback paths and give to all paths
        for id in ids:
            # Make the paths of length max_lookback
            if self.method == 'max':
                data = self.pad_start(df.loc[id].values, 1)
                id_paths = [data[max(0, i - self.lookback):i] for i in range(2, data.shape[0] + 1)]
            elif self.method == 'fixed':
                # Get the id data and pad with the start value
                data = self.pad_start(df.loc[id].values, self.lookback)
                id_paths = [data[max(0, i):i+self.lookback + 1] for i in range(len(data) - self.lookback)]
            elif self.method == 'mean':
                data = self.pad_start(df.loc[id].values, 1)
                id_paths = []
                for i in range(2, data.shape[0] + 1):
                    id_paths.append(self.mean_start(data[0:i], lookback=self.lookback))

            all_paths.extend(id_paths)

        # Final path if submission
        if self.last_only:
            all_paths = [all_paths[-1]]
        all_paths = deepcopy(all_paths)
        return all_paths


class FeatureSaverMixin(BaseEstimator, TransformerMixin):
    """ Mixin for generic feature computation class

    This allows for a load from save check before computation.

    Must specify in new class:
        - save_loc: Save location and name to store the data.
        - transform_func: The standard transformation function.
    """
    def __init__(self, recompute=False, force_compute=False):
        """
        :param recompute: Recompute and save again.
        :param force_compute: Force computation but do not save.
        """
        self.recompute = recompute
        self.force_compute = True if recompute else force_compute
        self.base_save_dir = DATA_DIR + '/processed/features'


    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_func'):
            raise TypeError('Class must take a transform_func method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, labels=None):
        return self

    @timeit
    def transform(self, df):
        """
        Loads if possible else saves. Here data will contain a pure numpy array and frame will be the indexed df.
        transform_func always returns an array but to use .loc must be converted to a frame when saved
        """
        # If submitting, no attempt to load from save
        if SUBMISSION:
            data = self.transform_func(df)
            return data

        if hasattr(self, 'force_compute'):
            if self.force_compute:
                frame = pd.DataFrame(index=df.index, data=self.transform_func(df))
                if self.recompute:
                    save_pickle(frame, self.save_loc)
                return frame

        # Otherwise attempt to load from save, else perform transform and save
        try:
            frame = load_pickle(self.save_loc)

            # Fail if the frame is smaller than the dataframe
            if frame.shape[0] < df.shape[0]:
                raise ValueError('The saved frame does not contain all the dataframe entries')

            # Reduce frame save to df if necessary.
            if frame.shape[0] > df.shape[0]:
                frame = frame.loc[df.index]

        except:
            frame = pd.DataFrame(index=df.index, data=self.transform_func(df))
            save_pickle(frame, self.save_loc)

        return frame.values


class SignaturesFromIdDataframe(FeatureSaverMixin):
    """ Extends a dataframe to include lookback window signature features

    Example:
        df = ExtendWithSignatureFeatures(columns=['ShockIndex', 'Resp'], lookback=4, sig_options=options).transform(df)

    This will compute lookback windows for each timepoint of each id in the dataframe with a lookback of 4 for the
    specified columns. It the computes the signatures with specified options and concatenates the results back into the
    main dataframe.
    """
    def __init__(self, columns, lookback=5, lookback_method='max', options={}, last_only=False):
        self.columns = columns
        self.lookback = lookback
        self.lookback_method = lookback_method
        self.options = options

        # For last only
        self.last_only = last_only

        # For saving
        self.save_loc = (DATA_DIR +
                         '/processed/features/signatures/columns={}_lookback={}_method={}_options={}.pickle'
                         .format(columns, lookback, lookback_method, options)
                         )

    def fit(self, df, labels=None):
        return self

    @timeit
    def transform_func(self, df):
        # Get the features we will compute signatures of
        df_features = df[self.columns]

        # Make the paths
        paths = MakePaths(lookback=self.lookback, method=self.lookback_method, last_only=self.last_only).transform(df_features)

        # Compute signatures and make into a dataframe
        transformer = SignatureTransformer(**self.options)
        signatures = transformer.fit_transform(paths)

        return signatures


class AddMoments(FeatureSaverMixin):
    """ Adds statistical moments over a lookback window to the dataframe """
    def __init__(self, moments=2, lookback=10, start=2, save_name=None, over_paths=True, last_only=False, **kwargs):
        self.moments = moments
        self.lookback = lookback
        self.start = start  # The moment to start at
        self.over_paths = over_paths

        # If last only, also dont run parallel code
        self.last_only = last_only
        self.parallel = False if last_only else True

        # Set save and file locations
        super().__init__(**kwargs)
        save_dir = self.base_save_dir + '/moments'
        if save_name is None:
            self.save_loc = (save_dir + '/moments={}_lookback={}.pickle'.format(moments, lookback))
        else:
            self.save_loc = (save_dir + '/moments={}_lookback={}_name={}.pickle'.format(moments, lookback, save_name))

    @staticmethod
    def compute_moment(path, i):
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # Mean of empty slice warning
        moment = (1 / (path.shape[0] - 1)) * ((path - np.nanmean(path, axis=0)) ** i).sum(axis=0).reshape(1, -1)
        return moment

    @timeit
    def transform_func(self, df):
        # Make the paths
        paths = MakePaths(lookback=self.lookback, method='fixed', last_only=self.last_only).transform(df)
        paths = deepcopy(paths)

        # Compute moments of each path
        all_moments = []
        for i in range(self.start, self.moments + 1):
            # Compute the moments for each path
            moments = np.array(basic_parallel_loop(self.compute_moment, [(path, i) for path in paths], parallel=False)).squeeze()

            # Make into a dataframe being careful for last_only method
            if self.last_only:
                moments = moments.reshape(1, -1)

            if len(moments.shape) == 1:
                moments = moments.reshape(-1, 1)

            # Store ready to concat
            all_moments.append(moments)

        # Make a full moments df
        all_moments = np.concatenate(all_moments, axis=1)

        return all_moments


class GetNumMeasurements(FeatureSaverMixin):
    """ Gets the number of measurements that were taken over some lookback window. """
    def __init__(self, lookback=7, last_only=False, force_compute=False, **kwargs):
        # Mixin init
        super().__init__(**kwargs)

        # Things
        self.lookback = lookback
        self.last_only = last_only
        self.force_compute = force_compute

        # Feature mixin
        self.save_loc = self.base_save_dir + '/num_measurements/num={}.pickle'.format(lookback)

    @staticmethod
    def num_measurements_taken(df, num):
        df = deepcopy(df)
        return df.rolling(num).apply(lambda x: x[-1] - x[0])

    def transform_func(self, df):
        df = deepcopy(df)
        # if self.last_only:
        #     counts_24hrs = self.num_measurements_taken(df, self.lookback).iloc[[-1]]
        # else:
        #     # Ugly but stange behaviour was happening before
        #     df_copy = deepcopy(df)
        #     for col in df_copy.columns:
        #         df_col = df_copy[[col]]
        #         measurement_nums = df_col.groupby('id').apply(self.num_measurements_taken, self.lookback)
        #         df_copy[col] = measurement_nums
        #     counts_24hrs = df_copy
        if self.last_only:
            counts_24hrs = self.num_measurements_taken(df_reduced, self.lookback).iloc[[-1]].values
        else:
            counts_24hrs = df.groupby('id').apply(self.num_measurements_taken, self.lookback).values
        return counts_24hrs


class GetRateOfLaboratorySampling():
    """ Divides the number of taken measurements by time to give the rate at which measurements have been sampled. """
    def __init__(self, last_only=False):
        self.last_only = last_only

    @timeit
    def transform(self, df):
        """ Finds any column with '_count' and divides by the corresponding val in ICULOS. """
        if not self.last_only:
            count_cols = [x for x in df.columns if '_count' in x]

            if self.last_only:
                df = df.iloc[[-1]]

            df[count_cols] = df[count_cols].values / df[['ICULOS']].values
            return df

        else:
            mask = np.argwhere([True if '_count' in x else False for x in df.columns]).reshape(-1)
            iculos_mask = np.argwhere([True if 'ICULOS' in x else False for x in df.columns]).reshape(-1)
            arr = df.iloc[[-1]].values
            arr[:, mask] = arr[:, mask] / arr[:, iculos_mask]
            return arr


class AddBaseline(BaseEstimator, TransformerMixin):
    """
    For a given shift and window size, gets the rolling mean of each feature from 'shift' number of timepoints. Then
    appends to a df or returns the features.
    """
    def __init__(self, shift=7, rolling_size=10):
        self.shift = shift
        self.rolling_size = rolling_size

    def fit(self, df, labels=None):
        return df

    def groupby_func(self, df):
        return df.shift(self.shift).rolling(self.rolling_size, min_periods=5).mean()

    def transform(self, df, return_full=True):
        # Compute rolling mean with some window shift and add to df
        rolling_mean = groupby_apply_parallel(df.groupby('id'), self.groupby_func)
        baseline_diff = df - rolling_mean
        baseline_diff.columns = [x + '_basediff' for x in df.columns]

        # Return appended to df or solo
        if return_full:
            return pd.concat([df, baseline_diff], axis=1)
        else:
            return baseline_diff


class AbsoluteSumOfChanges(FeatureSaverMixin):
    def __init__(self, columns=False, lookback=10):
        self.columns = columns
        self.lookback = lookback

        # Save location for Mixin
        self.save_loc = DATA_DIR + '/processed/features/absolute_sum/columns={}_lookback={}.pickle'.format(columns, lookback)

    @staticmethod
    def abs_sum_func(df, lookback):
        abs_changes = (df.shift(1) - df).abs()
        return abs_changes.rolling(lookback).sum()

    def transform_func(self, df):
        # Reduce if only applying to specific cols
        if self.columns is not False:
            data = df[self.columns]
        else:
            data = df

        # Apply to each id and return
        changes_frame = groupby_apply_parallel(data.groupby('id'), self.abs_sum_func, self.lookback)
        changes_frame.columns = [x + '.abs_changes' for x in df.columns]

        return changes_frame


class GetStatistic(FeatureSaverMixin):
    """ Gets a specified stastic through a rolling window approach. """
    def __init__(self, statistic='std', columns=None, lookback=7, roll=2, last_only=False, force_compute=False, recompute=False):
        self.columns = columns
        self.lookback = lookback
        self.statistic = statistic
        self.roll = roll
        self.last_only = last_only
        self.force_compute = force_compute
        self.recompute = recompute

        # Save location for Mixin
        self.save_loc = DATA_DIR + '/processed/features/basic_statistics/stat={}_lookback={}_roll={}_columns={}.pickle'.format(statistic, lookback, roll, str(columns)[0:100])

    def func(self, df):
        if self.statistic == 'min':
            return df.rolling(self.lookback, min_periods=self.roll).min()
        elif self.statistic == 'max':
            return df.rolling(self.lookback, min_periods=self.roll).max()
        elif self.statistic == 'std':
            return df.rolling(self.lookback, min_periods=self.roll).std()
        elif self.statistic == 'median':
            return df.rolling(self.lookback, min_periods=self.roll).median()
        elif self.statistic == 'mean':
            return df.rolling(self.lookback, min_periods=self.roll).mean()

    def transform_func(self, df):
        if self.last_only is False:
            return df.groupby('id').apply(self.func)
        else:
            if self.statistic == 'min':
                return df.iloc[-self.lookback:].min()
            elif self.statistic == 'max':
                return df.iloc[-self.lookback:].max()
            elif self.statistic == 'std':
                return df.iloc[-self.lookback:].std()
            elif self.statistic == 'median':
                return df.iloc[-self.lookback:].median()
            elif self.statistic == 'mean':
                return df.iloc[-self.lookback:].mean()


class GetMeanReducedFeatures():
    def __init__(self, columns=None, last_only=None, recompute_all=False):
        self.columns = columns
        self.last_only = last_only
        self.recompute_all = recompute_all

    def get_saved_df(self):
        mean_reduced = load_pickle(DATA_DIR + '/processed/dataframes/full_mean_reduced_data.pickle')
        mean_reduced = mean_reduced[[x + '_mean_reduced' for x in self.columns]]
        return mean_reduced

    def mean_reduce(self, df):
        # Make numpy
        rolling_mean = df.rolling(100000000000, min_periods=1).mean().values
        return pd.DataFrame(index=df.index, data=(df.values - rolling_mean), columns=df.columns)

    def transform(self, df):
        if self.last_only:
            df = df[self.columns]
            df_reduced = pd.DataFrame(index=df.index, data=self.mean_reduce(df).iloc[[-1]], columns=df.columns)
            df_reduced.columns = [x + '_mean_reduced' for x in df_reduced.columns]
        elif self.recompute_all:
            # Proceed as normal if not last only
            mean_reduced_data = df.groupby('id').apply(self.mean_reduce)
            mean_reduced_data.columns = [x + '_mean_reduced' for x in mean_reduced_data.columns]
            save_pickle(mean_reduced_data, DATA_DIR + '/processed/dataframes/full_mean_reduced_data.pickle')
            return mean_reduced_data
        else:
            mean_reduced_data = self.get_saved_df()
            return mean_reduced_data


def nan_interpolate_and_ffill(path):
    """ Where data exists, linearly interpolates between, ffill is used at the end """
    # TODO this method again but use lininterp from prev data if it exists
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    # If all rows have at least one value, use lininterpolate or ffill bfill
    if not any([x == path.shape[0] for x in np.isnan(path).sum(axis=0)]):
        nans, x = nan_helper(path)
        path[nans] = np.interp(x(nans), x(~nans), path[~nans])
    return path


if __name__ == '__main__':
    df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
    a = GetMeanReducedFeatures(recompute_all=True).transform(df[list(df.columns[0:7]) + ['SOFA']])

