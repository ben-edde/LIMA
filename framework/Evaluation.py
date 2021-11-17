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

def evaluate(model, X, y, h):
    def get_TS_cv(k=10, horizon=0):
        """
        2 ways to split:
        * one testing point per horizon (each horizon is then independent)
        * series of horizon (large horizon include smaller horizon: overlap)
        Simply looking at single point error may not be good, as the performance should include a forecast on series. The overlapping should be kept.
        For t=0, take single point
        For t>=1, take series
        """
        if horizon == 0:
            return TimeSeriesSplit(
                n_splits=k,
                gap=horizon - 1,
                test_size=1,
            )
        return TimeSeriesSplit(
            n_splits=k,
            gap=0,
            test_size=horizon,
        )
    try:
        cv=get_TS_cv(horizon=h)
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
        evaluation_result={'h':h,'mae':[mae.mean()],'rmse':[rmse.mean()],'mape':[mape.mean()],'descriptions':msg}
        return evaluation_result
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)


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
    evaluation_result={'h':horizon,'mae':[mae],'rmse':[rmse],'mape':[mape],'descriptions':""}
    return evaluation_result

# does not make sense to future data to predict past: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
# def get_random_cv(k=10):
#     return RepeatedKFold(n_splits=k, random_state=42)