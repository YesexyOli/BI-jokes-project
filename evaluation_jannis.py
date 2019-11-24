import numpy as np
import pandas as pd
import prediction_web as pre_w
"""
Evaluate prediction by comparing ratted values with predicted values. And sum up 
the difference. 
"""
def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df


def drop_elements(rating, percent_to_drop):
    holey = rating.copy()
    num_row = rating.shape[0]
    num_col = rating.shape[1]
    element_in_table = num_row * num_col
    num_element_to_drop = int(element_in_table * percent_to_drop)
    index_to_drop_row = np.random.randint(0, num_row, num_element_to_drop)
    index_to_drop_col = np.random.randint(0, num_col, num_element_to_drop)
    for i in range(num_element_to_drop):
        holey.iloc[index_to_drop_row[i], index_to_drop_col[i]] = 99
    return holey, index_to_drop_row, index_to_drop_col


def calc_evaluation(full, prediction, index_to_drop_row, index_to_drop_col):
    abs_error = 0
    for i in range(len(index_to_drop_row)):
        field_full = full.iloc[index_to_drop_row[i], index_to_drop_col[i]]
        field_prediction = prediction.iloc[index_to_drop_row[i], index_to_drop_col[i]]
        abs_error = np.abs(field_full-field_prediction)
    return abs_error, abs_error/(2*len(index_to_drop_col))


if __name__ == '__main__':
    start_col_jokes = 3
    df_center = load_data('data/centers_norm')
    df_full = load_data('data/full_rated_jokes_norm.csv')
    df_rating = df_full.iloc[:, 3:]
    #np_rating_sol = np.array(df_rating)
    df_holey, row_droped, col_droped = drop_elements(df_rating, 0.7)
    df_holey = pd.concat([df_full.iloc[:, :3], df_holey], axis=1)

    df_prediction = df_holey.apply(pre_w.make_perdiction_one_row, 1,
                                  args=[df_center, start_col_jokes])
    total_error, error_per_prediction = calc_evaluation(df_full.iloc[:, 3:], df_prediction.iloc[:, 3:], row_droped, col_droped)
    print('The total difference is {}, the mean difference per prediction is {}'.format(total_error, error_per_prediction))