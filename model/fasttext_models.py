import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import (TimeSeriesSplit, cross_validate)
from sklearn.svm import SVR,LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, ARDRegression, SGDRegressor, ElasticNet, Lars, Lasso, GammaRegressor, TweedieRegressor, PoissonRegressor, Lasso
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('Fasttext_models')
exp.observers.append(FileStorageObserver('runs'))

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)

def get_TS_cv(k=10, horizon=3):
    return TimeSeriesSplit(
        n_splits=k,
        gap=0,
        test_size=horizon,
    )


def evaluate(model, X, y, cv):
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
    print(f"""
    Forecast Error ({k}-fold cross-validated performance):
    {model.__class__.__name__}:
    MAE = {mae.mean():.3f} +/- {mae.std():.3f}
    RMSE = {rmse.mean():.3f} +/- {rmse.std():.3f}
    MAPE = {mape.mean():.3f} +/- {mape.std():.3f}
    """)
    return cv_results

@exp.automain
def main():
    df_news_price = pd.read_pickle(
        f"{os.environ['LIMA_HOME']}/embedding/WTI_Spot_n_RedditNews_2008-06-09_2016-07-01_fasttext.pkl"
    )
    df_news_price = df_news_price[::-1]
    X = np.array(df_news_price.News_fasttext.to_numpy().reshape(-1).tolist())
    y = df_news_price.Price.to_numpy().reshape(-1)
    ts_cv = get_TS_cv()

    res = evaluate(model=LinearSVR(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=DecisionTreeRegressor(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=KernelRidge(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=Ridge(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=BayesianRidge(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=ARDRegression(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=LinearRegression(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=SGDRegressor(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=ElasticNet(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=Lars(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=Lasso(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=GammaRegressor(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=TweedieRegressor(), X=X, y=y, cv=ts_cv)

    res = evaluate(model=PoissonRegressor(), X=X, y=y, cv=ts_cv)
