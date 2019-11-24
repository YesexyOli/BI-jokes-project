"""
The attempt to use a neural network to make predictions
"""
from keras import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename, index_col=None, header=0)
    return df

df_full = load_data('data/full_data.csv')
df_rating = df_full.iloc[:, 3:]
np_rating_sol = np.array(df_rating)
num_row = np_rating_sol.shape[0]
num_col = np_rating_sol.shape[1]
element_in_table =  num_row * num_col
num_element_to_drop = int(element_in_table * 0.5)
jannis_vec = np.array([[1,-2,1,1,-1,-2,-1,-1,-1,1,99,99,99,99,-2,-2,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99]])
#jannis_vec.reshape((1,num_col))
jannis_vec[jannis_vec == 99] = 0
dummy_vec = np.array([[-1.0,99,-1.0,-2.0,-2.0,-2.0,1.0,1.0,2.0,2.0,2.0,2.0,0.0,1.0,2.0,-1.0,0.0,0.0,-1.0,1.0,2.0,-1.0,0.0,-2.0,-2.0,-1.0,1.0,0.0,1.0,0.0,0.0,2.0,0.0,2.0,2.0,0.0,0.0,2.0,2.0,2.0,2.0,2.0,0.0,-2.0,-1.0,2.0,2.0,1.0,1.0,2.0,-2.0,0.0,0.0,0.0,2.0,-1.0,0.0,0.0,2.0,2.0,1.0,1.0,0.0,-1.0,-2.0,2.0,2.0,0.0,0.0,0.0,2.0,2.0,2.0,2.0,-2.0,2.0,-2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,-2.0,2.0,-2.0,2.0,-2.0,2.0,2.0,2.0,0.0,2.0,2.0,2.0,-2.0,2.0
]])
dummy_vec[dummy_vec == 99]=0
index_to_drop_row = np.random.randint(0, num_row, num_element_to_drop)
index_to_drop_col = np.random.randint(0, num_col,  num_element_to_drop)
np_rating_input = np_rating_sol.copy()
for i in range(num_element_to_drop):
    np_rating_input[index_to_drop_row[i], index_to_drop_col[i]] = 0


#(x_train, _), (x_test, _) = mnist.load_data()
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(num_col,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='tanh')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(num_col, activation='tanh')(encoded)

# this model maps an input to its reconstruction
#autoencoder = Model(input_img, decoded)
autoencoder = Sequential()
autoencoder.add(Dense(100, input_dim=100, activation='relu'))
autoencoder.add(Dense(50, activation='relu'))
autoencoder.add(Dense(20, activation='relu'))
autoencoder.add(Dense(50, activation='relu'))
autoencoder.add(Dense(100, activation='linear'))



autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
"""
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
"""
autoencoder.fit(np_rating_input, np_rating_sol,
                epochs=100,
                batch_size=200,
                shuffle=True,
                verbose=1)
                #validation_data=(np_rating_input, np_rating_sol))
print(autoencoder.predict(jannis_vec))
print(autoencoder.predict(dummy_vec))