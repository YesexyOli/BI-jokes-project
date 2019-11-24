"""
discretization of the ratings in the values -1, -0.5, 0 , 0.5, 1
adding column USER_ID and NEED_TO_CHANGE
Runtime behaviour:
    23:05 min for changing all values if iterating field for field
    0:30 min with np.apply_along_axis

"""

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocessing(filename_raw, filename_boiled):
    df = pd.read_excel(filename_raw, index_col=None, header=None)
    column_name = ['Rated_Jokes']
    for i in range(0, df.shape[1] - 1, 1):
        column_name.append('Joke_{}'.format(i))
    df.columns = column_name
    #df = df.loc[df:, (df != 99).any(axis=0)]
    tqdm.pandas()
    result = df.progress_apply(change_values, axis=1)
    result.insert(loc=0, column='USER_ID', value=np.arange(len(df)))
    result.insert(loc=1, column='NEED_TO_CHANGE', value=1)
    result.to_csv(filename_boiled, index=None)

def change_values(row):
    for column in range(1, row.shape[0], 1):
        value = float(row[column])
        if - 10 <= value < -6:
            row[column] = -1
        if - 6 <= value < -2:
            row[column] = -0.5
        if - 2 <= value < 2:
            row[column] = 0
        if 2 <= value < 6:
            row[column] = 0.5
        if 6 <= value <= 10:
            row[column] = 1
    return row


preprocessing(sys.argv[1], sys.argv[2])
# data/jester_data_100_raw.csv result/jester_data_100_boiled.csv
# data/test_raw.csv data/test_boiled.csv