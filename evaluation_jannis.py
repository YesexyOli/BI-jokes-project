import numpy as np
import pandas as pd
from tqdm import tqdm

import prediction_web as pre_w
"""
Evaluate prediction by comparing ratted values with predicted values. And sum up 
the difference. 
"""
def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=None)
    return df


def drop_elements(rating, percent_to_drop=0.5):
    holey = rating.copy()
    num_row = rating.shape[0]
    num_col = rating.shape[1]
    element_in_table = num_row * num_col
    num_element_to_drop = int(element_in_table * percent_to_drop)
    index_to_drop_row = np.random.randint(0, num_row, num_element_to_drop)
    index_to_drop_col = np.random.randint(0, num_col, num_element_to_drop)
    indecies = zip(index_to_drop_row, index_to_drop_col)
    size_before_unique = num_element_to_drop
    indecies = list(set(indecies))
    size_after_unique = len(indecies)
    for tuple in indecies :
        holey.iloc[tuple[0], tuple[1]] = 99
    return holey, indecies , size_before_unique - size_after_unique


def calc_evaluation(full, prediction, indecies):
    abs_error = 0
    error = 0
    x_square =0
    for tuple in indecies:
        field_full = full.iloc[tuple[0], tuple[1]]
        field_prediction = prediction.iloc[tuple[0], tuple[1]]
        abs_error = abs_error + np.abs(field_full-field_prediction)
        error = error + field_full-field_prediction
        x_square = x_square + np.abs(field_full-field_prediction)*np.abs(field_full-field_prediction)
    mean = error/len(indecies)
    mean_square = mean*mean
    mean_x_square = x_square/len(indecies)

    abs_error = 0
    error = 0
    x_square = 0
    s=0
    for tuple in indecies:
        field_full = full.iloc[tuple[0], tuple[1]]
        field_prediction = prediction.iloc[tuple[0], tuple[1]]
        abs_error = abs_error + np.abs(field_full - field_prediction)
        error = error + field_full - field_prediction
        x_square = x_square + np.abs(field_full - field_prediction) * np.abs(field_full - field_prediction)
        s = (field_full - field_prediction - mean)*(field_full - field_prediction - mean)
    return abs_error, mean, np.sqrt(mean_x_square - mean_square), np.square(s/(len(indecies)-1))


if __name__ == '__main__':
    start_col_jokes = 3
    df_center = pd.read_csv('data/full_rated_jokes_norm.csv', index_col=None, header=0).iloc[:, 3:]
    df_full = pd.read_csv('data/full_rated_jokes_norm_sub.csv', index_col=None, header=0)
    df_rating = df_full.iloc[:, 3:]

    df_holey, indices_deleted, num_double = drop_elements(df_rating, percent_to_drop=0.5)
    df_holey = pd.concat([df_full.iloc[:, :3], df_holey], axis=1)
    tqdm.pandas()
    df_prediction = df_holey.progress_apply(pre_w.make_perdiction_one_row, axis=1,
                                  args=[df_center, start_col_jokes])
    total_error, mean_error, standart_derivation, standart_derivation_2 = calc_evaluation(df_full.iloc[:, 3:], df_prediction.iloc[:, 3:], indices_deleted)
    print('{} Elements where deletet'.format(num_double))
    print('The total difference is {}, the error per field is {} the mean is {} with a standart derivation of {} and {} '.format(total_error, total_error/len(indices_deleted), mean_error, standart_derivation, standart_derivation_2))