import datetime
import logging
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.layers import Reshape, MaxPool3D, Bidirectional, ConvLSTM2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Conv2D
from keras.layers import Dropout
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso
import random
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

HOME = os.environ['LIMA_HOME']


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

        names += [f'{data.columns[j]}(t-{i})' for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{data.columns[j]}(t)' for j in range(n_vars)]
        else:
            names += [f'{data.columns[j]}(t+{i})' for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# set random seed
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
h = 1
past = 10

df_WTI = pd.read_csv(f"{HOME}/dev/features/WTI_4C.csv")
df_WTI.Date = pd.to_datetime(df_WTI.Date)
df_WTI.set_index("Date", inplace=True)
df_WTI = df_WTI[["CLC4", "CLC3", "CLC2", "CLC1"]]
df_features = pd.read_pickle(
    f"{HOME}/dev/features/RedditNews_2008-06-09_2016-07-01_4_features.pkl")
df_features.index = pd.to_datetime(df_features.index)
df_Xy = pd.concat([df_features, df_WTI], axis=1, join="inner")

# preserve original price for inverting prediction
df_original_price = df_Xy[["CLC1"]].shift(h).dropna()
# 1st order DIFF
df_Xy = df_Xy.diff().dropna()

# shift back $past days
df_Xy = series_to_supervised(df_Xy, past, h)

df_original_price = df_original_price[df_original_price.index.isin(
    df_Xy.index)]

# remove current day features for forecast
for each in df_Xy.columns[:-1]:
    if "(t)" in each:
        df_Xy.drop(each, axis=1, inplace=True)
raw_X = df_Xy.to_numpy()[:, :-1]
y = df_Xy.to_numpy()[:, -1].reshape(-1, 1)

X_scaler = MinMaxScaler(feature_range=(0.1, 1))
raw_X = X_scaler.fit_transform(raw_X)
y_scaler = MinMaxScaler(feature_range=(0.1, 1))
y = y_scaler.fit_transform(y)

estimator = Ridge()
selector = RFE(estimator, n_features_to_select=30, step=1)
selector = selector.fit(raw_X, y.ravel())
# df_Xy.columns[:-1][selector.get_support()]
X = raw_X[:, selector.get_support()]



train_X = X
train_y = y

model = Sequential()
model.add(Reshape((1, train_X.shape[-1])))
model.add(GRU(300, dropout=0.33, input_shape=(1, train_X.shape[-1])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(train_X,
                    train_y,
                    epochs=50,
                    batch_size=100,
                    verbose=0,
                    shuffle=False)

# export trained model
model.save(f"{HOME}/dev/models/GRU.model")
joblib.dump(X_scaler, f"{HOME}/dev/models/feature_scaler(110).joblib")
joblib.dump(y_scaler, f"{HOME}/dev/models/label_scaler(1).joblib")
joblib.dump(selector, f"{HOME}/dev/models/feature_selector(30).joblib")
