import logging
import os
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('ARIMA_benchmark')
exp.observers.append(FileStorageObserver('ARIMA_benchmark'))

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format='%(asctime)s %(name)s %(filename)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)


def TS_evaluate(model,y, h):
    """
    Evaluation function for TS model using k-fold cross validation. Forecast horizon define size of test set.

    Args:
        y       : uniseries dependence variable
        h (int) : forecast horizon, determine num of out-sample forecast made by model, and the corresponding test set.
    """
    def get_TS_cv(k=10, horizon=1):
        return TimeSeriesSplit(
            n_splits=k,
            gap=0,
            test_size=horizon,
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

    try:
        cv = get_TS_cv(horizon=h)
        df_forecast_error = pd.DataFrame(
            columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
        for train_idx, test_idx in cv.split(y):
            train_y = y[train_idx].reshape(-1, 1)
            test_y = y[test_idx].reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(train_y)
            train_y=scaler.transform(train_y)
            test_y=scaler.transform(test_y)
            model.fit(train_y)
            pred_y = model.predict(h)
            forecast_error = evaluate_series(test_y, pred_y, h)
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
        MAE = {mae.mean():.3f} +/- {mae.std():.3f}
        RMSE = {rmse.mean():.3f} +/- {rmse.std():.3f}
        MAPE = {mape.mean():.3f} +/- {mape.std():.3f}
        """
        print(msg)
        logging.info(msg)
        evaluation_result = {
            'h': h,
            'mae': [mae.mean()],
            'rmse': [rmse.mean()],
            'mape': [mape.mean()],
            'descriptions':
            f"ARIMA(order={model.order}, seasonal_order={model.seasonal_order}, AIC={model.aic()}, BIC={model.bic()})"
        }
        return evaluation_result
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)


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

    for h in range(1, 2):
        model=pm.ARIMA(order=(2, 1, 3),seasonal_order=(1, 0, 1, 6))
        result = TS_evaluate(model=model,y=df_price.Price.to_numpy(), h=h)
        df_result = df_result.append(pd.DataFrame(result), ignore_index=True)
    df_result.to_csv("arima_results.csv", mode="a", index=False, header=False)
