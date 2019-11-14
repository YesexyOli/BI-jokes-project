import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool




class BestList:
    def __init__(self, max_size):
        self.max_size = max_size
        self.corr_value = []
        self.row_index = []
        self.smallest_index = None
        self.smallest_value = None

    def add(self, value, r_index):
        if len(self.corr_value) == 0:
            self.corr_value.append(value)
            self.smallest_index = 0
            self.smallest_value = value
            self.row_index.append(r_index)

        elif len(self.corr_value) < self.max_size:
            self.corr_value.append(value)
            self.row_index.append(r_index)
            if self.smallest_value > value:
                self.smallest_value = value
                self.smallest_index = self.find_smallest_value()

        elif len(self.corr_value) == self.max_size:
            if self.smallest_value < value:
                self.corr_value.pop(self.smallest_index)
                self.row_index.pop(self.smallest_index)
                self.corr_value.append(value)
                self.row_index.append(r_index)
                self.smallest_index = self.find_smallest_value()

        if len(self.corr_value) > self.max_size:
            raise ValueError('Size of the Bestlist is bigger than allowed. This is a programming mistake!')

    def find_smallest_value(self):
        return np.array(self.corr_value).argmin()

    def give_back(self):
        return self.corr_value, self.row_index


def most_relevant_user(matrix, user_row, number_of_user, col_joke_begin,
                       col_joke_end):
    """
    :param matrix:
    :param user_row:
    :param number_of_user:
    :return:
    """
    best_user = BestList(number_of_user)
    user_row_np = (np.array(user_row)[col_joke_begin: col_joke_end])
    user_row_np[user_row_np == 99] = 0
    for index, row in matrix.iterrows():
        row_np = np.array(row)[col_joke_begin: col_joke_end]
        row_np[row_np == 99] = 0
        corr_value = np.dot(user_row_np, row_np) / \
                     (np.linalg.norm(user_row_np) * np.linalg.norm(row_np))
        best_user.add(corr_value, row.USER_ID)
    return best_user.give_back()


def calculate_rating(matrix, user_row, joke_index, number_most_similar_user, col_joke_begin,
                     col_joke_end):
    """

    :param matrix, matrix with user which rated joke for which recommendation is made:
    :param user_row:
    :param joke_index:
    :param number_most_similar_user:
    :param col_joke_begin:
    :param col_joke_end:

    Find the most similar user between the all user who rated this joke
    calculate weighted average value
    :return:
        prediction for one field in matrix
    """
    index_of_row_not_containing_joke = matrix.index[matrix.iloc[:, joke_index] == 99]
    relevant_matrix = matrix.drop(index_of_row_not_containing_joke)
    corr_values, user_ids = most_relevant_user(relevant_matrix, user_row, number_most_similar_user,
                                               col_joke_begin, col_joke_end)
    rating_total = 0
    corr_value_total = 0
    for i in range(len(user_ids)):
        user_id = user_ids[i]
        row = matrix.loc[matrix['USER_ID'] == user_id]
        rating_total += corr_values[i] * row.iloc[0, joke_index]
        corr_value_total += np.abs(corr_values[i])
    if corr_value_total == 0:
        return 0
    return rating_total / corr_value_total


def make_prediction(df, Threshold_min_rated_jokes, number_most_similar_user, start_index_jokes, end_index,
                    number_random_user_to_compare):
    """
    :param df: df has the form:
    #rated_jokes, joke_1 , ... , joke_n, UserID, Need_to_change
    :param Threshold_min_rated_jokes minimal value of rated jokes
     a user have to be rated to use for prediction :
    :param number_most_similar_user number of user for calculating the average:


    :var

    df_replaced contains same information as df, but the value for unknown ratings is chaged from 99 to 0
    df_relevant contains all rows from df_replaced, where the user rated more jokes than Threshold_min_rated_jokes


    :return:
    df_prediction matrix with all redicted values
    df_true_values_and_prediction matrix with all entries
    """
    df_replaced = df.copy()  # .concat([df.iloc[:, 0], df.iloc[:, 1:-2].replace(99, np.NaN), df.iloc[:, -2:]], axis=1)
    index_of_row_under_threshold = df_replaced.index[df_replaced.iloc[:, 0] < Threshold_min_rated_jokes]
    df_relevant = df_replaced.drop(index_of_row_under_threshold)
    end_index_jokes = df.shape[1] + end_index
    index_col_NEED_TO_CHANGE = df.columns.get_loc("NEED_TO_CHANGE")

    df_prediction = df.apply(make_perdiction_one_row, 1, args=(df_relevant,
                                                               start_index_jokes, end_index_jokes,
                                                               number_most_similar_user, number_random_user_to_compare,
                                                               index_col_NEED_TO_CHANGE))
    df_prediction["NEED_TO_CHANGE"] = 0
    return df_prediction


def make_perdiction_one_row(row, df_relevant, start_index_jokes, end_index_jokes, number_most_similar_user,
                            number_random_user_to_compare, index_col_NEED_TO_CHANGE):
    row_unchanged = row.copy()
    row_prediction = np.full(row.shape[0], np.nan)
    if row['NEED_TO_CHANGE'] == 1:
        for column in range(start_index_jokes, end_index_jokes, 1):
            if row.iloc[column] == 99:
                if number_random_user_to_compare > df_relevant.shape[0]:
                    number_random_user_to_compare = df_relevant.shape[0]
                    if number_random_user_to_compare == 0:
                        print("Random user table is empty, probably Threshold to high!")
                df_random = df_relevant.ix[
                    np.random.choice(df_relevant.index, number_random_user_to_compare, replace=False)]
                rating = calculate_rating(df_random, row_unchanged, column, number_most_similar_user,
                                          start_index_jokes, end_index_jokes)
                row.iloc[column] = round(rating, 0)
                row_prediction[column] = round(rating, 0)

    return row


def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df


def multiprozessing(input):
    subset = input[0]
    Threshold_min_rated_jokes = input[1]
    num_most_similar_user = input[2]
    start_col_jokes = input[3]
    end_col_jokes = input[4]
    num_random_user_to_compare = input[5]
    df_pred = make_prediction(subset, Threshold_min_rated_jokes, num_most_similar_user,
                                  start_col_jokes, end_col_jokes, num_random_user_to_compare)
    print ("Subset begin with line {} from {} is prozessed".format(input[6], input[7]))
    return df_pred

# data/test_boiled.csv 2 3 1 -2 TRUE 10
if len(sys.argv) != 10:
    print(
        "Script expect 9 Parameter 'path_to_csv', 'Threshold_min_rated_jokes', 'number_most_similar_user', start_index_jokes,"
        " end_index_jokes, if results should be print, num_random_user_to_compare, size_of_sub_table, number prozesses")
else:
    data_matrix = load_data(sys.argv[1])
    Threshold_min_rated_jokes = int(sys.argv[2])
    num_most_similar_user = int(sys.argv[3])
    start_col_jokes = int(sys.argv[4])
    end_col_jokes = int(sys.argv[5])
    num_random_user_to_compare = int(sys.argv[7])
    begin_subset = 0

    input_multiprocessing = []
    for i in range(0, data_matrix.shape[0], int(sys.argv[8])):
        if i + int(sys.argv[8]) > data_matrix.shape[0]:
            begin_subset = data_matrix.shape[0] - int(sys.argv[8])
        else:
            begin_subset = i
        subset = data_matrix.iloc[begin_subset:begin_subset + int(sys.argv[8]), :]
        input_multiprocessing.append([subset, Threshold_min_rated_jokes, num_most_similar_user,
                                      start_col_jokes, end_col_jokes, num_random_user_to_compare, begin_subset, data_matrix.shape[0]])

    with Pool(int(sys.argv[9])) as p:
        result_multi_proz = p.map(multiprozessing, input_multiprocessing)

    result = pd.DataFrame(columns = data_matrix.columns)

    for sub_result in result_multi_proz:
       result = pd.concat([result, sub_result])
    result.to_csv('results/predicated_values.csv', index=None)
    if sys.argv[6]:
        print("df: ", result, sep="\n\n", end="\n\n")


        # data/jesterfinal_boiled.csv 2 3 1 -2 TRUE 10 10 6
        # data/test_boiled.csv 2 10 1 -2 FALSE 50 100 1

