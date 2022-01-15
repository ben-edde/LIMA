import pmdarima as pm
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
from statsmodels.api import tsa
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('RW_benchmark')
exp.observers.append(FileStorageObserver('RW_benchmark'))

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format='%(asctime)s %(name)s %(filename)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
HOME = os.environ['LIMA_HOME']

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

@exp.automain
def main():
    h=1
    past=10

    df_y = pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/WTI_Spot_2008-06-09_2016-07-01.csv")
    df_y.set_index("Date",inplace=True)
    # df_y=df_y.diff()
    df_sen=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_sentiment_daily.pkl")
    df_sen.set_index("Date",inplace=True)
    # df_event=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_event_tuples_fasttext_daily.pkl")
    # df_X=pd.merge(df_sentiment,df_event,on="Date")
    df_geo=pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_daily_GeoIdx.csv")
    df_geo.set_index("Date",inplace=True)
    df_topic=pd.read_pickle(f"{HOME}/data/reuse/RedditNews_WTI/features/RedditNews_2008-06-09_2016-07-01_4_topic_daily.pkl")
    df_Xy=pd.concat([df_sen,df_geo,df_topic,df_y],axis=1)
    original_price=df_Xy.Price

    
    y=df_Xy.to_numpy()[:,-1].reshape(-1, 1)
    try:
        cv = get_TS_cv()
        df_forecast_error = pd.DataFrame(
            columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
        for train_idx, test_idx in cv.split(y):
            
            train_y = y[train_idx]
            test_y = y[test_idx]
            
            # normalize features
            
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaler.fit(train_y)

            
            train_y=y_scaler.transform(train_y)
            test_y=y_scaler.transform(test_y)

            
            print(f"train_y: {train_y.shape}\ntest_y:{test_y.shape}")
            pred_y=[]
            for i in range(test_y.shape[0]):
               
                train_data=train_y.tolist()+test_y[:i].tolist()
                print(f"train_data: {len(train_data)}")

                model = tsa.UnobservedComponents(train_data, "rwalk")
                fitted_model = model.fit()
                forecasts = fitted_model.forecast(1)
                pred_y.append(forecasts)
            
            inverted_pred_y = y_scaler.inverse_transform(pred_y)
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

    evaluation_result["descriptions"]="RandomWalk"
    df_result = pd.DataFrame(evaluation_result)
    df_result["time"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    df_result = df_result[['time', 'descriptions', 'h', 'mae', 'rmse', 'mape']]
    df_result.to_csv(f"{HOME}/results/experiment_results.csv",
                    mode="a+",
                    index=False,
                    header=False)

