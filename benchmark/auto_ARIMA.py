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


def evaluate(y_true, y_pred, horizon):
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return pd.DataFrame({
        f"rmse_{horizon}": [rmse],
        f"mae_{horizon}": [mae],
        f"mape_{horizon}": [mape]
    })


def get_TS_cv(k=10, horizon=3):
    return TimeSeriesSplit(
        n_splits=k,
        gap=0,
        test_size=horizon,
    )


@exp.automain
def main():
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
                                  seasonal=True,
                                  m=m)
            msg = f"Parameters: \n{model.get_params()}"
            print(msg)
            logging.info(msg)
            print(model.summary())
            for h in range(1, 4):
                forecasts = model.predict(h)
                result = evaluate(df_price.iloc[test_idx[:h]].Price, forecasts,
                                  h)
                msg = f"forecast(h={h}): \n{result}"
                print(msg)
                logging.info(msg)
