"""
The attemp to use joke labels to make prediction better not finished yet
"""

import pandas as pd
import numpy as np


def calc_content(row, df_jokes_content, start_col_jokes):
    df_user_profil = pd.Series([0 for _ in range(len(df_jokes_content.columns))], index=df_jokes_content.columns)
    total=0
    for i in range(start_col_jokes, row.size[0], 1):
        df_user_profil += row.iloc[i]*df_jokes_content.iloc[i,:]
        total +=  row.iloc[i]
    df_user_profil_norm = df_user_profil/total
    return pd.concat([row, df_user_profil_norm], axis=0)


if __name__ == '__main__':
    df_jokes_content = pd.read_csv("data/BIjokes.csv", index_col=0, header=0)
    df_jokes_content.replace('Y', 1, inplace=True)
    df_jokes_content.replace(np.nan, 0, inplace=True)
    # df.to_csv('data/full_data.csv', index=None)
    df_jokes_content.drop(['content'], axis=1, inplace=True)

    df = pd.read_csv("data/full_data_boiled.csv", index_col=None, header=0)
    user_content = df.apply(calc_content, args=[df_jokes_content, 3],axis=1)

    print(df)
