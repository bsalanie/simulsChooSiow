import itertools
import pandas as pd

"""
given a dictionary, return a pandas DataFrame grid of all combinations 
"""


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())
