import os
import dill, pickle
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import json
from sty import fg, bg
import pandas as pd


def save_pickle(obj, filename, use_dill=False, protocol=4):
    """ Basic pickle/dill dumping """
    _create_folder_if_not_exist(filename)
    with open(filename, 'wb') as file:
        if not use_dill:
            pickle.dump(obj, file, protocol=protocol)
        else:
            dill.dump(obj, file)


def load_pickle(filename, use_dill=False):
    """ Basic pickle/dill loading function """
    with open(filename, 'rb') as file:
        if not use_dill:
            obj = pickle.load(file)
        else:
            obj = dill.load(file)
    return obj


def _create_folder_if_not_exist(filename):
    """ Makes a folder if the path does not already exist """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def load_json(filename):
    with open(filename) as file:
        obj = json.load(file)
    return obj


def ppprint(message, color='blue', start=False):
    """
    Color prints a message, generally used to print info when running sake

    :param message: The message to print
    :param color: Color to print
    :param start: Set to true for a new line start and dash line break
    :return: None (prints message to console)
    """
    # initialise the color
    init_col = eval('fg.li_{}'.format(color))

    # Extra emphasis if a major function
    if start:
        message = '\n' + message + '\n' + '-' * 100

    # sty message
    to_print = init_col + message + fg.rs

    # Print to terminal
    print(to_print)


def basic_parallel_loop(func, *args, parallel=True):
    """ Basic parallel computation loop.

    :param func (function): The function to be applied.
    :param *args (list): List of arguments [(arg_1_1, ..., arg_n_1), (arg_1, 2), ..., (arg_k_n)]. Each tuple of args is
    fed into func
    :return: Results from the function in a list
    """
    # Create pooling and run
    # pool = Pool(processes=num_cpus)
    # results = pool.starmap(func, args[0])

    if parallel is True:
        results = Parallel(n_jobs=cpu_count())(delayed(func)(*a) for a in args[0])
    else:
        results = []
        for a in args[0]:
            results.append(func(*a))

    return results


def groupby_apply_parallel(grouped_df, func, *args):
    with Pool(cpu_count()) as p:
        ret_list = p.starmap(func, [(group, *args) for name, group in grouped_df])
    return pd.concat(ret_list)


if __name__ == '__main__':
    def mult(a, b): return a*a*b
    args = [(i, i) for i in range(10000)]
    basic_parallel_loop(mult, args)

