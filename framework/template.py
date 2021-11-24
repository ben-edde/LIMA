import datetime
import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('exp')
exp.observers.append(FileStorageObserver('runs'))

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
        cv = get_TS_cv()
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

# @exp.automain
# def main():
    # HOME = os.environ['LIMA_HOME']
    # df_result = pd.DataFrame(
    #     columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
    # df_Xy=df_Xy[::-1]
    # X = np.array(df_Xy.X.to_numpy().reshape(-1).tolist())
    # y = df_Xy.y.to_numpy().reshape(-1)
    # for h in range(1,6):
    #     result = evaluate(model, X, y, h=h)
    #     result["descriptions"] = ""
    #     df_result = df_result.append(pd.DataFrame(result), ignore_index=True)
    # df_result["time"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    # df_result = df_result[['time', 'descriptions', 'h', 'mae', 'rmse', 'mape']]
    # df_result.to_csv(f"{HOME}/results/results.csv",
    #                  mode="a",
    #                  index=False,
    #                  header=False)
