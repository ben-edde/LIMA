
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
from keras.layers import Dropout
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
HOME = os.environ['LIMA_HOME']

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



h=1
past=10

df_y = pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/WTI_Spot_2008-06-09_2016-07-01.csv")
df_y.Date=pd.to_datetime(df_y.Date)
df_y.set_index("Date",inplace=True)

df_Price1=pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/Cushing_OK_Crude_Oil_Future_Contract_1.csv")
df_Price1.columns=["Date","Price1"]
df_Price1.Date=pd.to_datetime(df_Price1.Date)
df_Price1.set_index("Date",inplace=True)

df_Price2=pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/Cushing_OK_Crude_Oil_Future_Contract_2.csv")
df_Price2.columns=["Date","Price2"]
df_Price2.Date=pd.to_datetime(df_Price2.Date)
df_Price2.set_index("Date",inplace=True)

df_Price3=pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/Cushing_OK_Crude_Oil_Future_Contract_3.csv")
df_Price3.columns=["Date","Price3"]
df_Price3.Date=pd.to_datetime(df_Price3.Date)
df_Price3.set_index("Date",inplace=True)

df_Price4=pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/Cushing_OK_Crude_Oil_Future_Contract_4.csv")
df_Price4.columns=["Date","Price4"]
df_Price4.Date=pd.to_datetime(df_Price4.Date)
df_Price4.set_index("Date",inplace=True)

df_sen=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_sentiment_daily.pkl")
df_sen.Date=pd.to_datetime(df_sen.Date)
df_sen.set_index("Date",inplace=True)

df_geo=pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_daily_GeoIdx.csv")
df_geo.Date=pd.to_datetime(df_geo.Date)
df_geo.set_index("Date",inplace=True)

df_topic=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_4_topic_daily.pkl")
df_topic.index=pd.to_datetime(df_topic.index)


df_Xy=pd.concat([df_sen,df_geo,df_topic,df_Price1,df_Price2,df_Price3,df_Price4,df_y],axis=1,join="inner")
# df_Xy=df_Xy.iloc[::-1]


# from statsmodels.tsa.stattools import grangercausalitytests
# grangercausalitytests(df_Xy[["Price1","Price"]],10)
# grangercausalitytests(df_Xy[["Price2","Price"]],10)
# grangercausalitytests(df_Xy[["Price3","Price"]],10)
# grangercausalitytests(df_Xy[["Price4","Price"]],10)


# df_Xy=df_Xy.diff().dropna()
from statsmodels.tsa.stattools import adfuller
for i in range(df_Xy.shape[1]):
    test_result=adfuller(df_Xy[df_Xy.columns[i]].to_numpy())
    if test_result[1]>0.05:
        print(f"{df_Xy.columns[i]}: {test_result[1]}")


original_price=df_Xy.Price
df_Xy=df_Xy.diff().dropna()
df_Xy=series_to_supervised(df_Xy,past,h)
df_Xy.columns


df_Xy.head()


# df_Xy=df_Xy.diff().dropna()
from statsmodels.tsa.stattools import adfuller
for i in range(df_Xy.shape[1]):
    test_result=adfuller(df_Xy[df_Xy.columns[i]].to_numpy())
    if test_result[1]>0.05:
        print(f"{df_Xy.columns[i]}: {test_result[1]}")


for each in df_Xy.columns[:-1]:
    if "(t)" in each:
        df_Xy.drop(each,axis=1,inplace=True)
raw_X=df_Xy.to_numpy()[:,:-1]
y=df_Xy.to_numpy()[:,-1].reshape(-1, 1)


# # granger
# raw_X=df_Xy[df_Xy.columns[:15]].to_numpy()
# raw_X.shape


from sklearn.feature_selection import mutual_info_regression,RFE,SelectFromModel,SequentialFeatureSelector,chi2,SelectKBest,f_regression,VarianceThreshold
from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.svm import LinearSVR,SVR
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,RandomForestRegressor,GradientBoostingRegressor
estimator = Lasso()
selector = RFE(estimator, n_features_to_select=10, step=1)
# selector=SelectFromModel(estimator)
# selector=SequentialFeatureSelector(estimator,n_features_to_select=7,direction='forward',n_jobs=-1)
# selector=SelectKBest(f_regression,k=7)
# selector=VarianceThreshold(3.21)
selector = selector.fit(raw_X, y.ravel())



# # mutual info
# # m_info=mutual_info_regression(raw_X, y.ravel())
# select_idx=m_info.argsort()[:5]
# df_selected_features=df_Xy[df_Xy.columns[select_idx]]
# df_Xy.columns[select_idx]




# RFE and MODEL
df_selected_features=df_Xy[df_Xy.columns[:-1][selector.get_support()]]
df_Xy.columns[:-1][selector.get_support()]



# from sklearn.decomposition import PCA,FastICA,FactorAnalysis
# # pca = PCA(n_components=7,svd_solver='full')
# decomposer = FactorAnalysis(n_components=7)
# X=decomposer.fit_transform(raw_X)
# X.shape



X=df_selected_features.to_numpy()
# X=raw_X
X.shape


from keras.layers import Reshape,MaxPool3D,Bidirectional




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
        train_X=train_X.reshape(train_X.shape[0],1,1,1,train_X.shape[-1])
        test_X=test_X.reshape(test_X.shape[0],1,1,1,test_X.shape[-1])
        print(f"train_X: {train_X.shape}\ntest_X:{test_X.shape}")

        model = Sequential()
        model.add(Bidirectional(ConvLSTM2D(300,(1,1),return_sequences=True)))
        # model.add(Conv2D(300,(1,1)))
        # model.add(Reshape((1,1,300)))
        # model.add(MaxPool2D((1,1)))
        model.add(Dropout(0.2))
        model.add(Reshape((1,600)))
        model.add(Bidirectional(GRU(100,dropout=0.33,return_sequences=True)))
        model.add(Bidirectional(GRU(50,dropout=0.33,return_sequences=True)))
        # model.add(GRU(300,dropout=0.33,input_shape=(train_X.shape[1],train_X.shape[-1])))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        history = model.fit(train_X, train_y, epochs=30, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=True)
        # # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        pred_y = model.predict(test_X)
    
        inverted_pred_y = y_scaler.inverse_transform(pred_y.reshape(-1, 1))
        inverted_pred_y=inverted_pred_y+original_price[test_idx].to_numpy()

        inverted_test_y = y_scaler.inverse_transform(test_y)  # should be same as testXy
        inverted_test_y=inverted_test_y+original_price[test_idx].to_numpy()
        
                    
        forecast_error = evaluate_series(inverted_test_y, inverted_pred_y, h)
        print(forecast_error)
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




df_forecast_error


evaluation_result["descriptions"]="Bidirectional_ConvLSTM2D_300_Drop_0.2_Bidirectional_GRU_100_Bidirectional_GRU_50; 1st DIFF+ RFE(Lasso,10)"
df_result = pd.DataFrame(evaluation_result)
df_result["time"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

df_result = df_result[['time', 'descriptions', 'h', 'mae', 'rmse', 'mape']]
df_result




# df_result.to_csv(f"{HOME}/results/experiment_results.csv",
#                     mode="a+",
#                     index=False,
#                     header=False)





