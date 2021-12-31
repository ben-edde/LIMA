import datetime
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


HOME = os.environ['LIMA_HOME']
df_y = pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/WTI_Spot_2008-06-09_2016-07-01.csv")
df_X=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_sentiment_daily.pkl")
df_Xy=pd.merge(df_X,df_y,on="Date")
df_Xy.set_index("Date",inplace=True)

total_size=df_Xy.shape[0]
train_size=int(total_size*0.6)


from matplotlib import pyplot
# values = df_Xy.to_numpy()
# # specify columns to plot

# # plot each column
# pyplot.figure()
# for i in range(df_Xy.shape[1]):
# 	pyplot.subplot(df_Xy.shape[1], 1, i+1)
# 	pyplot.plot(values[:, i])
# 	pyplot.title(df_Xy.columns[i], y=0.5, loc='right')

# pyplot.show()


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


df_Xy=series_to_supervised(df_Xy,3,1)
df_Xy.drop(df_Xy.columns[-7:-1],axis=1,inplace=True)
trainXy=df_Xy.iloc[:train_size].to_numpy()
testXy=df_Xy.iloc[train_size:].to_numpy()


from sklearn.preprocessing import MinMaxScaler
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(trainXy)

trainXy=scaler.transform(trainXy)
testXy=scaler.transform(testXy)

train_X,train_y=trainXy[:,:-1], trainXy[:,-1]
test_X,test_y=testXy[:,:-1], testXy[:,-1]

# reshape to 3D for LSTM
train_X=train_X.reshape(train_X.shape[0],1,train_X.shape[-1])
test_X=test_X.reshape(test_X.shape[0],1,test_X.shape[-1])

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=50, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()


# make a prediction
yhat = model.predict(test_X)
yhat.shape

# reshape test_X back to 2D for inverse
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_X_yhat = scaler.inverse_transform(np.concatenate((test_X,yhat), axis=1))
inv_yhat=inv_X_yhat[:,-1]

# invert scaling for actual
test_y = test_y.reshape(-1, 1)

inv_Xy = scaler.inverse_transform( np.concatenate((test_X,test_y), axis=1))  # should be same as testXy
inv_y = inv_Xy[:,-1]



from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# RMSE
rmse = mean_squared_error(inv_y, inv_yhat, squared=False)
# MAE
mae = mean_absolute_error(inv_y, inv_yhat)
# MAPE
mape = mean_absolute_percentage_error(inv_y, inv_yhat)

print('Test RMSE: %.3f' % rmse)

print(f"""
RMSE: {rmse}
MAE: {mae}
MAPE: {mape}
""")