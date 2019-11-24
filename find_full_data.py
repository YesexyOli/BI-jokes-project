"""
Contain some hepling methods:
 - returning user which ratted all jokes
 - normalizing user vectors
 - cluster User with k-means Algorithm and hopfully get in this way a good representation
"""
import pandas as pd
# import hierarchical clustering libraries
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm


def get_full_rated_jokes():
    df = pd.read_csv("data/jester_data_100_boiled.csv",  header=0)
    df = df.loc[df['Rated_Jokes'] == 100, :]
    df_norm = df.apply(normalize, axis=1, args=[3])
    df_norm.to_csv('data/full_rated_jokes_norm.csv', index=None)

def normalize(row, start_index_jokes):
    """
    :param row:
    :return: normalized row and mean
    calculate mean of ech row and subtract it from all values which are not 99 (unrated joke)
    """
    row_jokes = row.iloc[start_index_jokes:]
    mean = (row_jokes.loc[row_jokes.values != 99]).mean()
    row_norm = row_jokes.apply(lambda x: x-mean if x != 99 else 99)
    return pd.concat([row.iloc[:start_index_jokes], row_norm])

def get_n_most_different_user(num_cluster):
    df = pd.read_csv("data/full_rated_jokes_norm.csv", header=0)
    df_jokes = df.iloc[:, 3:]
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(df_jokes)
    centers = kmeans.cluster_centers_
    df_centers = pd.DataFrame(centers, columns=df.columns[3:])
    df_centers.drop_duplicates(inplace=True)
    if df_centers.shape[0] != num_cluster:
        droped_lines = num_cluster - df_centers.shape[0]
        print("{} out of {} centers where drop because they where duplicates".format(droped_lines, num_cluster))
    df_centers.to_csv('data/centers_norm', index=False)

def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df

if __name__ == '__main__':
    #get_full_rated_jokes()
    get_n_most_different_user(50)
    #df = load_data('data/full_rated_jokes.csv')
    #tqdm.pandas()
    #df_norm = df.progress_apply(normalize, axis=1, args=[3])
    #df_norm.to_csv('data/full_rated_jokes_norm.csv', index=False, header=False)
