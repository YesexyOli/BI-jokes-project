import sys
import numpy as np
import pandas as pd
import random

# load preprocessed data file

df = pd.read_csv("test.csv", index_col=None, header=0)

# copy datafile

df.to_csv("./copied_test.csv")
df_copied = pd.read_csv("copied_test.csv")

# Randomly choose components
# Store chosen components to the list named "original_data"

'''
If value is 99(empty), re-choose.
Else, store it to list 
'''

i = 0
j = 0


def random_choose():
    global i
    i = random.randint(0, 5)
    global j
    j = random.randint(0, 5)


def store_to_list():
    random_choose()

    while df.iloc[i, j] == 0 :
        if df.iloc[i, j] != 99:
            original_data = [[i, j, df.iloc[i][j]]]
    else:
        random_choose()


# Delete chosen components in the copied file
'''
Change value to 99(empty)
'''


def delete_data():
    df_copied.iloc[i][j] = 99


# Prediction processing of deleted components
'''
Should connect with 'Prediction.py'

'''

store_to_list()
delete_data()

# Store predicted value to the list named "predicted_data"

predicted_data = [[i, j, df_copied.iloc[i][j]]]

# Compare "original_data" and "predicted_data"

error_value = abs(df.iloc[i][j] - df_copied.iloc[i][j])
total_err_value = 0
total_err_value += error_value

# Error val
print(total_err_value)
