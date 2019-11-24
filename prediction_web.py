"""Because the  recommander for the web page the webpage musst be
very fast not all users are compared with each other instead each user is
compared to a representiv subsample of the users."""

import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df


def calc_ratings(row, center, start_col_jokes ):
    return row

def similarity_func(v, u):
    counter = 0
    abs_u = 0
    abs_v = 0
    for i in range(len(u)):
        if u[i] != 99:
            counter += u[i]*v[i]
            abs_u += np.abs(u[i])
            abs_v += np.abs(v[i])
    if abs_u != 0 and abs_v != 0:
        return counter/(abs_u*abs_v)
    else: return 0


def make_perdiction_one_row(row, df_center, start_index_jokes):
    """
    :param row:
    :param df_center: already centered 
    :param start_index_jokes: 
    :return: 
    """
    row_norm, mean = normalize(row, start_index_jokes)
    row_norm_with_prediction = row_norm.copy()
    sim_vec = df_center.apply(similarity_func, args=[row_norm[start_index_jokes:]], axis=1)
    if row_norm['NEED_TO_CHANGE'] == 1:
        for column in row.index[3:]:
            if row_norm.loc[column] == 99:
                rating = np.dot(sim_vec, df_center[column].replace(99,0))
                total_weight = np.sum(np.abs(sim_vec))
                if total_weight == 0:
                    row_norm_with_prediction.loc[column] = 0
                else:
                    row_norm_with_prediction.loc[column] = rating / total_weight
            else: row_norm_with_prediction.loc[column] = -99
        row_norm_with_prediction['NEED_TO_CHANGE'] = 0
    row_denorm_with_pred = denormalize(row_norm_with_prediction, start_index_jokes, mean)
    return row_denorm_with_pred



def normalize(row, start_index_jokes):
    """
    :param row:
    :return: normalized row and mean
    calculate mean of ech row and subtract it from all values which are not 99 (unrated joke)
    """
    row_jokes = row.iloc[start_index_jokes:]
    mean = (row_jokes.loc[row_jokes.values != 99]).mean()
    row_norm = row_jokes.apply(lambda x: x-mean if x != 99 else 99)
    return pd.concat([row.iloc[:start_index_jokes], row_norm]), mean

def denormalize(row, start_index_jokes, mean ):
    """
    :param row:
    :param start_index_jokes:
    :return: row summed up with mean
    """
    row_jokes = row.iloc[start_index_jokes:]
    row_norm = row_jokes.apply(lambda x: x + mean if x != 99 else 99)
    return pd.concat([row.iloc[:start_index_jokes], row_norm])


def find_best_predictions(row, start_index_jokes):
    jokes = row.iloc[start_index_jokes:].sort_values(ascending=False)
    result = pd.concat([pd.Series(row['USER_ID'], index=['USER_ID']),
                        pd.Series(list(jokes.index))])
    return result


while True:
    df_ratings = load_data(sys.argv[1])
    df_center = load_data(sys.argv[2])
    output_file = sys.argv[3]
    start_col_jokes = int(sys.argv[4])
    num_prozess = int(sys.argv[5])
    print_boolean = sys.argv[6] == 'True'

    begin_subset = 0
    tqdm.pandas()
    if print_boolean:
        print(" Make Prediction per line.")
    df_prediction = df_ratings.progress_apply(make_perdiction_one_row, axis=1,
                                  args=[df_center, start_col_jokes])
    if print_boolean:
        print(" Find best Jokes: ")
    df_best_jokes = df_prediction.progress_apply(find_best_predictions, 1, args=[start_col_jokes])

    df_best_jokes.to_csv(output_file, index=False, header=False)
