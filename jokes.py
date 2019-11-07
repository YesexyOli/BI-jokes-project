from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split


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


def most_relevant_user(matrix, user_row, number_of_user):
    best_user = BestList(number_of_user)
    user_row_np = np.array(user_row)[1:-1]
    for index, row in matrix.iterrows():
        row_np = np.array(row)[1:-1]
        corr_value = np.dot(user_row_np, row_np) / \
                     (np.linalg.norm(user_row_np) * np.linalg.norm(row_np))
        best_user.add(corr_value, row.USER_ID)
    return best_user.give_back()


def calculate_rating(matrix, user_row, joke_index, number_most_similar_user):
    index_of_row_containing_joke = matrix.index[matrix.iloc[:, joke_index] == 0]
    relevant_matrix = matrix.drop(index_of_row_containing_joke)  # .drop(matrix[matrix.USER_ID != user_id])
    corr_values, user_ids = most_relevant_user(relevant_matrix, user_row, number_most_similar_user)
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


def make_prediction(df, Threshold_min_rated_jokes, number_most_similar_user):
    df_prediction = df.copy()
    df_true_values_and_prediction = df.copy()
    df_replaced = pd.concat([df.iloc[:, 0], df.iloc[:, 1:-1].replace(99, 0), df.iloc[:, -1]], axis=1)
    index_of_row_under_threshold = df_replaced.index[df_replaced.iloc[:, 0] < Threshold_min_rated_jokes]
    df_relevant = df_replaced.drop(index_of_row_under_threshold)
    start_index_jokes = 1
    end_index_jokes = df.shape[1] - 1
    for row in tqdm(range(df.shape[0])):
        user_row = df_replaced.iloc[row, :].copy()
        for column in range(start_index_jokes, end_index_jokes, 1):
            if df.iloc[row, column] == 99:
                rating = calculate_rating(df_relevant.ix[np.random.choice(df_relevant.index, 10)], user_row, column, number_most_similar_user)
                df_prediction.iloc[row, column] = rating
                df_true_values_and_prediction.iloc[row, column] = rating
            else:
                df_prediction.iloc[row, column] = 99
    return df_prediction, df_true_values_and_prediction


def preprocessing(filename):
    df = pd.read_csv(filename, index_col=None, header=None)
    df['USER_ID'] = np.arange(len(df))
    df.to_csv('matrix_with_user_column.csv', index=None)

def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df

#preprocessing('jesterfinal.csv')
#train, test = train_test_split(df, test_size=0.2)
#data_matrix = load_data('test.csv')
data_matrix = load_data('matrix_with_user_column.csv')
Threshold_min_rated_jokes = 2
number_most_similar_user = 2
df_pred, df_all_values = make_prediction(data_matrix, Threshold_min_rated_jokes, number_most_similar_user)
print(df_pred)