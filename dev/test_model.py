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

def evaluate_series(y_true, y_pred, horizon):
    """
    Some models (like ARIMA) may not support cross_validate(), compare the forecasting result directly
    Args:
        y_true: y of test set
        y_pred: y of prediction
        horizon: forecast horizon

    Returns:
        DataFrame: single row DF with 3 metrics wrt horizon
    """
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2=r2_score(y_true, y_pred)
    forecast_error = {
        'h': horizon,
        'mae': [mae],
        'rmse': [rmse],
        'mape': [mape],
        'r2':[r2],
        'descriptions': ""
    }
    return forecast_error


# set random seed
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)

h = 1
past = 10

df_WTI= pd.read_csv(f"{HOME}/dev/features/WTI_4C.csv")
df_WTI.Date=pd.to_datetime(df_WTI.Date)
df_WTI.set_index("Date",inplace=True)
df_WTI=df_WTI[["CLC4","CLC3","CLC2","CLC1"]]
df_features= pd.read_pickle(f"{HOME}/dev/features/RedditNews_2008-06-09_2016-07-01_4_features.pkl")
df_features.index=pd.to_datetime(df_features.index)

df_Xy=pd.concat([df_features,df_WTI],axis=1,join="inner")
df_Xy=df_Xy[-30:]
# 1st order DIFF
df_original_price=df_Xy[["CLC1"]].shift(h).dropna()
df_Xy=df_Xy.diff().dropna()

# shift back $past days
df_Xy=series_to_supervised(df_Xy,past,h)
df_Xy.columns
df_original_price=df_original_price[df_original_price.index.isin(df_Xy.index)]
# remove current day features for forecast
for each in df_Xy.columns[:-1]:
    if "(t)" in each:
        df_Xy.drop(each,axis=1,inplace=True)
raw_X=df_Xy.to_numpy()[:,:-1]
y=df_Xy.to_numpy()[:,-1].reshape(-1, 1)
trained_model = keras.models.load_model(f"{HOME}/dev/models/GRU.model")
trained_X_scaler=joblib.load(f"{HOME}/dev/models/feature_scaler(110).joblib")
trained_y_scaler=joblib.load(f"{HOME}/dev/models/label_scaler(1).joblib")
trained_feature_selector=joblib.load(f"{HOME}/dev/models/feature_selector(30).joblib")
test_X=trained_X_scaler.transform(raw_X)
test_X=test_X[:,trained_feature_selector.get_support()]
pred_y=trained_model.predict(test_X)
inverted_pred_y=trained_y_scaler.inverse_transform(pred_y)+df_original_price.to_numpy()
test_y=y
inverted_test_y=test_y+df_original_price.to_numpy()
assert (df_WTI.loc[df_Xy.index]["CLC1"].to_numpy()==inverted_test_y.ravel()).all()
forecast_error = evaluate_series(inverted_test_y, inverted_pred_y, h)
print(forecast_error)






