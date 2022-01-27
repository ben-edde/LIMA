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
import pymongo
import random
import string
import fasttext
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

HOME = os.environ['LIMA_HOME']



def clean(text: str) -> list:
    """
    clean text with tokenization; stemming; removing stop word, punctuation, number, and empty string.

    Args:
        text (str): text

    Returns:
        list: cleaned text as list of tokenized str
    """

    # to list of token
    text = word_tokenize(text)

    # stemming and convert to lower case if not proper noun: punctuation and stop word seem to help POS tagging, remove them after stemming
    word_tag = pos_tag(text)
    porter = PorterStemmer()
    text = [
        porter.stem(each[0])
        if each[1] != "NNP" and each[1] != "NNPS" else each[0]
        for each in word_tag
    ]

    # remove stop word: it seems stemming skip stop word; OK to remove stop word after stemming;
    stop_word = set(stopwords.words('english'))
    text = [each for each in text if not each in stop_word]

    # remove punctuation
    text = [
        each.translate(str.maketrans('', '', string.punctuation))
        for each in text
    ]
    # text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text) # if using re

    # convert number to <NUM>
    text = ["<NUM>" if each.isdigit() else each for each in text]

    # remove empty string
    text = [each for each in text if each != ""]

    return text

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = data.copy()
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

def get_TS_cv(k=5, test_size=None):
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
tf.random.set_seed(seed_value)
h = 1
past = 25

# [markdown]
# # load metrics


from influxdb_client import InfluxDBClient
client= InfluxDBClient.from_config_file(f"{HOME}/dev/DB/influxdb_config.ini")
query_api = client.query_api()
df_WTI = query_api.query_data_frame("""
from(bucket: "dummy")
  |> range(start: 2011-04-01, stop: 2019-04-01)
  |> filter(fn: (r) => r["_measurement"] == "WTI") 
  |> filter(fn: (r) => r["type"] == "closing_price") 
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> drop(columns: ["_start", "_stop"])
""")
df_WTI=df_WTI[["_time","CLC4","CLC3","CLC2","CLC1"]]
df_WTI.columns=["Date","CLC4","CLC3","CLC2","CLC1"]
df_WTI.set_index("Date",inplace=True)
df_WTI.index=df_WTI.index.map(lambda each: each.date())
df_WTI.index=pd.to_datetime(df_WTI.index)
client.close()


month=[each.month for each in df_WTI.index]
day=[each.day for each in df_WTI.index]
day_in_week=[each.weekday() for each in df_WTI.index]
df_dt=pd.DataFrame()
df_dt["month"]=month
df_dt["day"]=day
df_dt["day_in_week"]=day_in_week
df_dt.index=df_WTI.index


df_WTI


# # Text Features


fasttext_model = fasttext.load_model(f"{HOME}/data/big/cc.en.300.bin")


df_sentiment=pd.read_pickle("df_sentiment_2type.pkl")
df_sentiment.shape


df_topic=pd.read_pickle("df_topic_2type.pkl")
df_topic.shape


df_geoidx=pd.read_pickle("df_geoidx_2type.pkl")
df_geoidx.shape


df_Xy = pd.concat([df_sentiment,df_topic, df_geoidx,df_WTI["CLC1"]], axis=1, join="inner")
print(df_Xy.shape)
df_Xy.head(1)


df_shifted = series_to_supervised(df_Xy.dropna(), past, h)
# remove current day features for forecast
for each in df_shifted.columns[:-1]:
    if "(t)" in each:
        df_shifted.drop(each, axis=1, inplace=True)
# add time feature without shift 
df_shifted=pd.concat([df_dt,df_shifted],axis=1).dropna()
raw_X = df_shifted.to_numpy()[:, :-1]
y =  df_shifted.to_numpy()[:, -1].reshape(-1, 1) 
# y = df_Xy[df_Xy.index.isin(df_selected.index)].to_numpy()[:, -1].reshape(-1, 1)
# y=df_WTI[df_WTI.index.isin(df_selected.index)]["CLC1"].to_numpy().reshape(-1, 1)
f"{raw_X.shape}   |{y.shape} | "

# [markdown]
# # Feature Selection


from sklearn.feature_selection import mutual_info_regression,RFE,RFECV,SelectFromModel,SequentialFeatureSelector,chi2,SelectKBest,f_regression,VarianceThreshold,r_regression
from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.svm import LinearSVR,SVR


estimator = Lasso(random_state=42)
selector = RFE(estimator,n_features_to_select=10,step=1)
scaled_raw_X=MinMaxScaler().fit_transform(raw_X)
selector = selector.fit(scaled_raw_X, y.ravel())


X = raw_X[:, selector.get_support()]
print(f"{X.shape} | {y.shape}")
df_shifted.columns[:-1][selector.get_support()]




# [markdown]
# # Model


from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, ARDRegression, SGDRegressor, ElasticNet, Lars, Lasso, GammaRegressor, TweedieRegressor, PoissonRegressor, Ridge, BayesianRidge
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from keras.layers import Reshape,MaxPooling2D,Bidirectional,ConvLSTM2D
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN 
from keras.layers import Conv2D,Conv3D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Flatten
from keras.models import Model
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(seed_value)


length=X.shape[0]
train_size=int(length*0.7)
step_size=1

train_X=X[:train_size]
train_y=y[:train_size,:]

test_X=X[train_size:]
test_y=y[train_size:,:]

X_scaler = MinMaxScaler()
X_scaler.fit(train_X)
train_X=X_scaler.transform(train_X)
test_X=X_scaler.transform(test_X)

# train_X=train_X.reshape(train_X.shape[0],step_size,train_X.shape[-1])
# test_X=test_X.reshape(test_X.shape[0],step_size,test_X.shape[-1])
print(f"train_X: {train_X.shape}\t   \t test_X:{test_X.shape}")
print(f"train_y: {train_y.shape}\t   test_y:{test_y.shape}")


all_X=np.concatenate([train_X,test_X])
lin_model=LinearRegression()
lin_model.fit(train_X,train_y.ravel())
linear_y=lin_model.predict(all_X)


evaluate_series(y,linear_y,1)


non_linear_y=y-linear_y.reshape(y.shape)
non_linear_y.shape


from matplotlib import pyplot as plt 
plt.plot(non_linear_y[250:300],'ob') 
plt.show()


train_non_linear_y=non_linear_y[:train_size,:]
test_non_linear_y=non_linear_y[train_size:,:]
f"{non_linear_y.shape}|{train_non_linear_y.shape}|{test_non_linear_y.shape}"


non_linear_y_scaler = MinMaxScaler(feature_range=(1, 100))
non_linear_y_scaler.fit(train_non_linear_y)
train_non_linear_y=non_linear_y_scaler.transform(train_non_linear_y)
test_non_linear_y=non_linear_y_scaler.transform(test_non_linear_y)
f"{train_non_linear_y.shape}|{test_non_linear_y.shape}"


# tf.keras.backend.clear_session()
ts_inputs = Input(shape=(train_X.shape[-1],))
ts_model=Reshape((step_size,train_X.shape[-1]))(ts_inputs)
ts_model=Bidirectional(GRU(1000,dropout=0.2 ,return_sequences=True))(ts_model)
ts_model= Dropout(0.4)(ts_model)
ts_model=Bidirectional(GRU(1000,dropout=0.2 ,return_sequences=True))(ts_model)
ts_model= Dropout(0.4)(ts_model)
ts_model=Bidirectional(GRU(1000,dropout=0.2 ,return_sequences=True))(ts_model)
ts_model= Dropout(0.4)(ts_model)
ts_model=Bidirectional(GRU(1000,dropout=0.2 ,return_sequences=True))(ts_model)
ts_model= Dropout(0.4)(ts_model)
ts_model=Bidirectional(GRU(1000,dropout=0.2 ,return_sequences=False))(ts_model)
ts_model= Dropout(0.4)(ts_model)
ts_model =Dense(1)(ts_model)
ts_model = Model(inputs=ts_inputs, outputs=ts_model)
# ts_model.compile(loss='mae', optimizer=Adam())
ts_model.compile(loss='log_cosh', optimizer=Adam(0.0002))
# ts_model.summary()


df_forecast_error = pd.DataFrame(
        columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
history = ts_model.fit(train_X, train_non_linear_y, epochs=200, batch_size=40, validation_data=(test_X, test_non_linear_y), verbose=1, shuffle=False)
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()


pred_non_linear_y = ts_model.predict(test_X)
# pred_non_linear_y.shape


inverted_pred_non_linear_y = non_linear_y_scaler.inverse_transform(pred_non_linear_y.reshape(test_non_linear_y.shape))
inverted_test_non_linear_y = non_linear_y_scaler.inverse_transform(test_non_linear_y)
test_linear_y=linear_y[train_size:].reshape(-1,1)
inverted_pred_non_linear_y=inverted_pred_non_linear_y+test_linear_y
inverted_test_non_linear_y=inverted_test_non_linear_y+test_linear_y

print(evaluate_series(test_y, inverted_pred_non_linear_y, h))


# evaluate_series(test_y, inverted_test_non_linear_y, h)





