import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedKFold, TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('Fasttext_LASSO_SVR')
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


def get_random_cv(k=10):
    return RepeatedKFold(n_splits=k, random_state=42)


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
        "WTI_Spot_n_RedditNews_2008-06-09_2016-07-01_fasttext.pkl")

    X = np.array(df_news_price.News_fasttext.to_numpy().reshape(-1).tolist())
    y = df_news_price.Price.to_numpy().reshape(-1)

    ts_cv = get_TS_cv()
    random_cv = get_random_cv()

    svr_ts_result = evaluate(model=SVR(), X=X, y=y, cv=ts_cv)
    svr_ran_result = evaluate(model=SVR(), X=X, y=y, cv=random_cv)

    lasso_TS_result = evaluate(model=Lasso(alpha=0.8), X=X, y=y, cv=ts_cv)
    lasso_ran_result = evaluate(model=Lasso(alpha=0.8), X=X, y=y, cv=random_cv)
