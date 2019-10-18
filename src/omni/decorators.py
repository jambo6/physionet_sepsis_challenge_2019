import time
import pandas as pd


def timeit(method):
    """ Record the time of a function """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r \n  %2.2f ms' % (method, (te - ts) * 1000))
        return result
    return timed



def numpy_method(method):
    """ Method for generic numpy arrays that can be simply applied to pandas series and dataframes """
    def wrapper(*args, **kwargs):
        if any([isinstance(x, pd.Series) or isinstance(x, pd.DataFrame) for x in args]):
            numpied = args[0].values
            output_np = method(numpied, **kwargs)
            output = pd.Series(index=args[0].index, data=output_np)
        else:
            output = method(*args, **kwargs)
        return output
    return wrapper
