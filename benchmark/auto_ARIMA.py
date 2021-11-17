import logging
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('ARIMA_benchmark')
exp.observers.append(FileStorageObserver('auto_pm'))

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format='%(asctime)s %(name)s %(filename)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)


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
    evaluation_result = {
        'h': horizon,
        'mae': [mae],
        'rmse': [rmse],
        'mape': [mape],
        'descriptions': ""
    }
    return evaluation_result


def get_TS_cv(k=10, horizon=3):
    return TimeSeriesSplit(
        n_splits=k,
        gap=0,
        test_size=horizon,
    )


@exp.automain
def main():
    HOME = os.environ['LIMA_HOME']
    df_result = pd.DataFrame(
        columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
    df_price = pd.read_csv("WTI_Spot_2008-06-09_2016-07-01.csv")
    df_price.Date = pd.to_datetime(df_price.Date, infer_datetime_format=True)

    df_price.dropna(inplace=True)
    # due to price data order from latest, need to invert order before forecast
    df_price = df_price.iloc[::-1]
    df_price.reset_index(inplace=True, drop=True)

    ts_cv = get_TS_cv()
    for train_idx, test_idx in ts_cv.split(df_price.Price):
        for m in [1, 3, 6, 9, 12]:
            model = pm.auto_arima(df_price.iloc[train_idx].Price,
                                  start_p=1,
                                  max_p=9,
                                  start_q=1,
                                  max_q=9,
                                  start_P=1,
                                  max_P=6,
                                  start_Q=1,
                                  max_Q=6,
                                  seasonal=True,
                                  m=m)
            msg = f"Parameters: \n{model.get_params()}"
            print(msg)
            logging.info(msg)
            print(model.summary())
            for h in range(1, 4):
                forecasts = model.predict(h)
                result = evaluate_series(df_price.iloc[test_idx[:h]].Price,
                                         forecasts, h)
                result[
                    "descriptions"] = f"ARIMA(order={model.order}, seasonal_order={model.seasonal_order}, AIC={model.aic()}, BIC={model.bic()})"
                df_result = df_result.append(pd.DataFrame(result),
                                             ignore_index=True)
                msg = f"forecast(h={h}): \n{result}"
                print(msg)
                logging.info(msg)
                df_result.to_csv(f"auto_arima_results.csv",
                                 mode="a",
                                 index=False,
                                 header=False)
