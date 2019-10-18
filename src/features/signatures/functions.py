"""
Functions used in compute signature transformer classes
"""
import functools
import numpy as np
import pandas as pd
# import esig.tosig as ts


def single_path_method(transform):
    """
    Converts a single path of dimension (n, m) into an array of dimension (1, n, m) so all signature methods from the
    signature_transformers file can be used. Squeezes out single dimensions at the end
    """
    @functools.wraps(transform)
    def reshape_and_run(*args, **kwargs):
        # Get X and reshape it if it has dimension 2, error if is not 2 or 3
        self, X = args
        # dimension = X.ndim
        # if dimension == 2:
        #     X = X.reshape(1, X.shape[0], X.shape[1])
        # elif dimension is not 3:
        #     raise Exception('Input X must have dimension 2 or 3')

        # Go ahead with the transform
        X_transformed = transform(self, X, **kwargs)

        return X_transformed

    return reshape_and_run


def signatures_from_dataframe(transform):
    """ Decorator that wraps the compute_signature fit transform method that allows use with a path dataframe """
    @functools.wraps(transform)
    def sort_dataframe(*args, **kwargs):
        # Get the arguments
        self, arg1 = args

        # Save index and reshape in ML form if is a path dataframe
        if isinstance(arg1, pd.DataFrame):
            df, idx = arg1, arg1.index
            n_timepoints = int(df.columns[0].split('_')[-1])    # TODO fix Hacky method used to get n_timepoints
            X = df.values.reshape(df.shape[0], n_timepoints, -1)
        else:
            X = arg1

        # Main transform method
        signatures = transform(self, X)

        # Remake as df if came in as a df
        if 'idx' in locals():
            signatures = pd.DataFrame(data=signatures, index=idx)

        return signatures

    return sort_dataframe


def signature_options_to_fname(signature_options):
    """ Takes a dictionary of signature info and turns into a filename """
    fname = '|'.join([name + '=' + str(item) for name, item in signature_options.items()])
    return fname

