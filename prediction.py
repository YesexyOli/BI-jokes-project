"""
script to predict unrated jokes with user-to-user filtering
include multiprocessing to make predictions faster
"""

import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import tqdm


def make_prediction(df, start_index_jokes):
    """
    :param df: df has the form:
    # UserID, Need_to_change, rated_jokes, joke_1 , ... , joke_n,
    :param start_index_jokes
    # col of first rated jokes at the moment 3

    :return:
    df_prediction matrix with all redicted values
    df_true_values_and_prediction matrix with all entries

    use user to user recommendation based on normalized data input
    """
    joke_rating_norm_mean = df.iloc[:, start_index_jokes:].apply(normalize, 1)
    joke_rating_norm = joke_rating_norm_mean.iloc[:, :-1]
    df_mean = joke_rating_norm_mean.iloc[:, -1]
    df_norm = pd.concat([df.iloc[:, :start_index_jokes], joke_rating_norm], axis=1)
    joke_rating = pd.concat([df['USER_ID'], joke_rating_norm.replace(99, 0)], axis=1)
    sim_matrix = pd.DataFrame(cosine_similarity(joke_rating), columns=df['USER_ID'],
                              index=df['USER_ID'])

    df_prediction = df_norm.apply(make_perdiction_one_row, 1,
                             args=(sim_matrix, joke_rating, df_norm, start_index_jokes))

    df_prediction = pd.concat([df_prediction, df_mean], axis=1)
    df_prediction_unnorm = df_prediction.apply(denormalize, 1, args=([start_index_jokes]))
    return df_prediction_unnorm.drop(['MEAN'], axis=1)


def make_perdiction_one_row(row, sim_matrix, joke_rating, df, start_index_jokes):
    """

    :param row: the rated jokes of one user
    :param sim_matrix:
    :param joke_rating:   matrix with the normalized joke ratings
    :param df:   matrix with the original values
    :param start_index_jokes:
    :return: row with added predictions for not rated jokes
    """
    row_with_prediction = row.copy()
    if row['NEED_TO_CHANGE'] == 1:
        for column in range(start_index_jokes, row.shape[0], 1):
            if row.iloc[column] == 99:
                user_id_who_rated_joke = df['USER_ID'][df.iloc[:, column] != 99]
                if len(user_id_who_rated_joke.values) > 0:
                    sim_vec = (sim_matrix[row['USER_ID']])[user_id_who_rated_joke.values]
                    col_name = row.index[column]
                    joke_vec = (joke_rating[col_name])[user_id_who_rated_joke.index]
                    rating = np.dot(sim_vec, joke_vec)
                    total_weight = np.sum(np.abs(sim_vec))
                    if total_weight == 0:
                        row_with_prediction.iloc[column] = 99
                    else:
                        row_with_prediction.iloc[column] = np.round(rating / total_weight, 1)
            else: row_with_prediction.iloc[column] = 99
    row_with_prediction['NEED_TO_CHANGE'] = 0
    return row_with_prediction


def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df


def similarity_func(u, v):
    return 1 / 1 + (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def multiprozessing(input):
    """

    :param input: list containing [subset of datamatrix, col of first rated joke]
    :return:
    """
    subset = input[0]
    start_col_jokes = input[1]
    df_pred = make_prediction(subset, start_col_jokes)
    return df_pred

def normalize(row):
    """

    :param row:
    :return: normalized row with mean appended
    calculate mean of ech row and subtract it from all values which are not 99 (unrated joke)
    """
    mean = (row.loc[row.values != 99]).mean()
    for col in range(row.shape[0]):
        if row.iloc[col] != 99:
            row.iloc[col] = row.iloc[col] - mean
    s_mean = pd.Series([mean], index=['MEAN'])
    row = row.append(s_mean)
    return row

def denormalize(row, input):
    """
    :param row:
    :param start_index_jokes:

    :return: row summed up with mean
    """
    start_index_jokes = input
    for col in range(start_index_jokes, row.shape[0]-1, 1):
        row.iloc[col] = row.iloc[col] + row.iloc[-1]
    return row



# data/test_boiled.csv 2 3 1 -2 TRUE 10
if len(sys.argv) != 7:
    print(
        "Script expect 6 Parameter 'path_to_csv', 'path_to_save_matrix_with_prediction ', start_index_jokes",
         "size_of_sub_table_to_calculate_predicition, number of prozesses, if results should be print")
else:
    data_matrix = load_data(sys.argv[1])
    output_file = sys.argv[2]
    start_col_jokes = int(sys.argv[3])
    size_subset = int(sys.argv[4])
    num_prozess = int(sys.argv[5])
    print_boolean = sys.argv[6] == 'True'

    begin_subset = 0

    input_multiprocessing = []
    for i in range(0, data_matrix.shape[0], size_subset):
        if i + size_subset > data_matrix.shape[0]:
            subset = data_matrix.iloc[i: data_matrix.shape[0], :]
        else:
            subset = data_matrix.iloc[i: i+size_subset:]
        input_multiprocessing.append([subset, start_col_jokes])

    result = pd.DataFrame(columns=data_matrix.columns)

    pool = Pool(num_prozess)
    for sub_result in tqdm.tqdm(pool.imap_unordered(multiprozessing, input_multiprocessing),
                                total=len(input_multiprocessing)):
        result = pd.concat([result, sub_result])

    result.to_csv(output_file, index=None)

    if print_boolean:
        print("df: ", data_matrix, sep="\n\n", end="\n\n")

    if print_boolean:
        print("result: ", result, sep="\n\n", end="\n\n")

        for i in range(result.shape[0]):
            line = result.iloc[i, start_col_jokes:]
            jokes = line.loc[(-5 < line) & (line < 5)]
            if jokes.shape[0] != 0:

                print("worst jokes:  ", str(jokes.nsmallest(5).index).partition("[")[2].partition("]")[0])
                print("Best jokes: ", jokes.nlargest(5))


"""
Example Input: 
data/test_boiled.csv results/test_pred.csv 3 3 6 TRUE
data/jester_data_100_boiled.csv results/jester_data_100_pred.csv 3 100 6 FALSE
data/jannis.csv results/jannis_pred.csv 3 100 6 FALSE

 data_matrix = load_data(sys.argv[1])
    output_file = sys.argv[2]
    start_col_jokes = int(sys.argv[3])
    size_subset = int(sys.argv[4])
    num_prozess = int(sys.argv[5])
    print_boolean = sys.argv[6test_raw.csv test_boiled.csv]
"""
