import sys
import pandas as pd
import numpy as np


"""
discretization of the ratings in the values -2, -1, 0 , 1, 2
adding column USER_ID and NEED_TO_CHANGE
Runtime behaviour: 
    23:05 min for changing all values if iterating field for field 
    0:30 min with np.apply_along_axis
"""

def preprocessing(filename_raw, filename_boiled):
    df = pd.read_csv(filename_raw, index_col=None, header=None)
    result = df.apply(change_values, axis=1)
    result['USER_ID'] = np.arange(len(df))
    result['NEED_TO_CHANGE'] = 1
    result.to_csv(filename_boiled, index=None)


def change_values(row):
    for column in range(1, row.shape[0], 1):
        value = float(row[column])
        if - 10 <= value < -6:
            row[column] = -2
        if - 6 <= value < -2:
            row[column] = -1
        if - 2 <= value < 2:
            row[column] = 0
        if 2 <= value < 6:
            row[column] = 1
        if 6 <= value <= 10:
            row[column] = 2
    return row


preprocessing(sys.argv[1], sys.argv[2])
