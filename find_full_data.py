import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv("data/jester_data_100_boiled.csv", index_col=None, header=0)
    df = df.loc[df['Rated_Jokes'] == 100, :]
    df.to_csv('data/full_data.csv', index=None)
    df_mean = df.mean().iloc[3:]

    print(df_mean.loc[df_mean < -0.8])