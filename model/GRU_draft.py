import datetime
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN 
from keras.layers import Conv2D 
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

HOME = os.environ['LIMA_HOME']

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def get_TS_cv(k=10, test_size=None):
    """
    ML models do not need to care about forecast horizon when splitting training and test set. Forecast horizon should be handled by feature preparation ([X_t-1,X_t-2...]). Actually repeated K-fold can also be used, but stick to TS split to align with TS_evaluate().
    """
    return TimeSeriesSplit(
        n_splits=k,
        gap=0,
        test_size=test_size,
    )

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
    forecast_error = {
        'h': horizon,
        'mae': [mae],
        'rmse': [rmse],
        'mape': [mape],
        'descriptions': ""
    }
    return forecast_error

def main():
    h=1
    past=3
    
    df_y = pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/WTI_Spot_2008-06-09_2016-07-01.csv")
    df_X=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_sentiment_daily.pkl")
    df_Xy=pd.merge(df_X,df_y,on="Date")
    df_Xy.set_index("Date",inplace=True)

    df_Xy=series_to_supervised(df_Xy,past,h)
    for each in df_Xy.columns[:-1]:
        if "(t)" in each:
            df_Xy.drop(each,axis=1,inplace=True)
    X=df_Xy.to_numpy()[:,:-1]
    y=df_Xy.to_numpy()[:,-1].reshape(-1, 1)
    try:        
        cv = get_TS_cv()
        df_forecast_error = pd.DataFrame(
            columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
        for train_idx, test_idx in cv.split(y):
            train_X = X[train_idx]
            test_X = X[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]
            
            # normalize features
            X_scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaler.fit(train_X)
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaler.fit(train_y)

            train_X=X_scaler.transform(train_X)
            test_X=X_scaler.transform(test_X)
            train_y=y_scaler.transform(train_y)
            test_y=y_scaler.transform(test_y)

            # reshape to 3D for RNN/LSTM/GRU
            train_X=train_X.reshape(train_X.shape[0],1,train_X.shape[-1])
            test_X=test_X.reshape(test_X.shape[0],1,test_X.shape[-1])
            print(f"train_X: {train_X.shape}\ntest_X:{test_X.shape}")

            model = Sequential()
            model.add(GRU(50, input_shape=(train_X.shape[1],train_X.shape[-1])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')

            history = model.fit(train_X, train_y, epochs=30, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)
            # # plot history
            # pyplot.plot(history.history['loss'], label='train')
            # pyplot.plot(history.history['val_loss'], label='test')
            # pyplot.legend()
            # pyplot.show()

            pred_y = model.predict(test_X)
        
            inverted_pred_y = y_scaler.inverse_transform(pred_y)
            

            inverted_test_y = y_scaler.inverse_transform(test_y)  # should be same as testXy
                        
            forecast_error = evaluate_series(inverted_test_y, inverted_pred_y, h)
            df_forecast_error = df_forecast_error.append(
                pd.DataFrame(forecast_error), ignore_index=True)
        mae = df_forecast_error["mae"]
        rmse = df_forecast_error["rmse"]
        mape = df_forecast_error["mape"]
        k = cv.get_n_splits()
        msg = f"""
        Forecast Error ({k}-fold cross-validation)
        X: {X.shape}
        y: {y.shape}
        h= {h}
        Model: {model.__class__.__name__}
        MAE = {mae.mean():.6f} +/- {mae.std():.3f}
        RMSE = {rmse.mean():.6f} +/- {rmse.std():.3f}
        MAPE = {mape.mean():.6f} +/- {mape.std():.3f}
        """
        print(msg)
        logging.info(msg)
        evaluation_result = {
            'h': h,
            'mae': [mae.mean()],
            'rmse': [rmse.mean()],
            'mape': [mape.mean()],
            'descriptions': [msg]
        }    
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)

