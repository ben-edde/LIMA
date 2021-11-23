"""
Evaluation functions and logging config. To be copied to experiment file instead of being imported to avoid import problem.
"""
import logging
import os
import pandas as pd
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format=
    '%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(message)s',
    level=logging.DEBUG)


def ML_evaluate(model, X, y, h):
    """
    Evaluation function for ML model using k-fold cross validation. Splitting data into training set and test set without concern of forecast horizon.

    Args:
        model   : ML model
        X       : shifted feature series wrt forecast horizon
        y       : uniseries dependence variable
        h (int) : no use, for documentation only
    """
    def get_TS_cv(k=10, test_size=None):
        """
        ML models do not need to care about forecast horizon when splitting training and test set. Forecast horizon should be handled by feature preparation ([X_t-1,X_t-2...]). Actually repeated K-fold can also be used, but stick to TS split to align with TS_evaluate().
        """
        return TimeSeriesSplit(
            n_splits=k,
            gap=0,
            test_size=test_size,
        )

    try:
        cv = get_TS_cv(
            test_size=int(len(y) * 0.2)
        )  # using 20% data as test set as suggested by https://otexts.com/fpp3/accuracy.html
        cv_results = cross_validate(model,
                                    X,
                                    y,
                                    scoring=[
                                        'neg_mean_absolute_error',
                                        'neg_root_mean_squared_error',
                                        'neg_mean_absolute_percentage_error'
                                    ],
                                    cv=cv,
                                    n_jobs=-1)
        mae = -cv_results["test_neg_mean_absolute_error"]
        rmse = -cv_results["test_neg_root_mean_squared_error"]
        mape = -cv_results["test_neg_mean_absolute_percentage_error"]
        k = cv.get_n_splits()
        msg = f"""
        Forecast Error ({k}-fold cross-validation)
        X: {X.shape}
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
            'descriptions': msg
        }
        return evaluation_result
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)


def TS_evaluate(model, y, h):
    """
    Evaluation function for TS model using k-fold cross validation. Forecast horizon define size of test set.

    Args:
        model   : TS model, only take y
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
            train_y = y[train_idx]
            test_y = y[test_idx]
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
            'descriptions': msg
        }
        return evaluation_result
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)


# does not make sense to future data to predict past for TS model: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
# def get_random_cv(k=10):
#     return RepeatedKFold(n_splits=k, random_state=42)
