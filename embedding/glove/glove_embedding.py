import datetime
import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge, SGDRegressor
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('GloVe_embedding')
exp.observers.append(FileStorageObserver('runs'))

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
        cv = get_TS_cv(horizon=h)
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
            'descriptions': [msg]
        }
        return evaluation_result
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)

@exp.automain
def main():
    HOME = os.environ['LIMA_HOME']
    df_result = pd.DataFrame(
        columns=['h', 'mae', 'rmse', 'mape', 'descriptions'])
    df_news_price = pd.read_pickle(
        f"{HOME}/embedding/glove/WTI_Spot_n_RedditNews_2008-06-09_2016-07-01_glove.pkl"
    )
    df_news_price = df_news_price[::-1]
    X = np.array(df_news_price.News_glove.to_numpy().reshape(-1).tolist())
    y = df_news_price.Price.to_numpy().reshape(-1)
    for h in range(4):
        result = evaluate(LinearSVR(), X, y, h=h)
        result["descriptions"] = "glove SVR"
        df_result = df_result.append(pd.DataFrame(result), ignore_index=True)
    for h in range(4):
        result = evaluate(Ridge(), X, y, h=h)
        result["descriptions"] = "glove Ridge"
        df_result = df_result.append(pd.DataFrame(result), ignore_index=True)
    for h in range(4):
        result = evaluate(SGDRegressor(), X, y, h=h)
        result["descriptions"] = "glove SGDRegressor"
        df_result = df_result.append(pd.DataFrame(result), ignore_index=True)

    df_result["time"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    df_result = df_result[['time', 'descriptions', 'h', 'mae', 'rmse', 'mape']]
    df_result.to_csv(f"{HOME}/embedding/glove/results.csv",
                     mode="a",
                     index=False,
                     header=False)
