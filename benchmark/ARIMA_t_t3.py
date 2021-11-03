import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('ARIMA_benchmark')
exp.observers.append(FileStorageObserver('runs'))


def evaluate(y_true, y_pred, horizon):
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return pd.DataFrame({
        "horizon": [horizon],
        "rmse": [rmse],
        "mae": [mae],
        "mape": [mape]
    })


@exp.config
def set_paramaters():
    p, d, q = 1, 1, 1


@exp.automain
def main(p, d, q):
    df_price = pd.read_csv("WTI_Spot_2008-06-09_2016-07-01.csv")
    df_price.Date = pd.to_datetime(df_price.Date, infer_datetime_format=True)

    df_price["Price_t+3"] = df_price.shift(0)["Price"]
    df_price["Price_t+2"] = df_price.shift(-1)["Price"]
    df_price["Price_t+1"] = df_price.shift(-2)["Price"]
    df_price["Price_t"] = df_price.shift(-3)["Price"]
    df_price.dropna(inplace=True)
    # due to price data order from latest, need to invert order before forecast
    df_price = df_price.iloc[::-1]
    df_price.reset_index(inplace=True, drop=True)

    arima_model = ARIMA(df_price['Price_t'], order=(p, d, q))
    arima_result = arima_model.fit()
    print(arima_result.summary())

    df_price["predicted_t"] = arima_result.predict(start=0,
                                                   end=df_price.index[-1] +
                                                   0).reset_index(drop=True)
    df_price["predicted_t+1"] = arima_result.predict(start=1,
                                                     end=df_price.index[-1] +
                                                     1).reset_index(drop=True)
    df_price["predicted_t+2"] = arima_result.predict(start=2,
                                                     end=df_price.index[-1] +
                                                     3).reset_index(drop=True)
    df_price["predicted_t+3"] = arima_result.predict(start=3,
                                                     end=df_price.index[-1] +
                                                     3).reset_index(drop=True)

    metric_t = evaluate(df_price["Price_t"],
                        df_price["predicted_t"],
                        horizon=0)
    metric_t1 = evaluate(df_price["Price_t+1"],
                         df_price["predicted_t+1"],
                         horizon=1)
    metric_t2 = evaluate(df_price["Price_t+2"],
                         df_price["predicted_t+2"],
                         horizon=2)
    metric_t3 = evaluate(df_price["Price_t+3"],
                         df_price["predicted_t+3"],
                         horizon=3)

    evaluation_result = metric_t.append([metric_t1, metric_t2, metric_t3],
                                        ignore_index=True)

    result_dict = evaluation_result.to_dict()
    [exp.log_scalar(each, result_dict[each]) for each in result_dict]
