import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedKFold, TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sacred import Experiment
from sacred.observers import FileStorageObserver
exp = Experiment('Event_GloVe_LASSO_SVR')
exp.observers.append(FileStorageObserver('runs'))

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
def main(datafile,alpha):
    alpha=float(alpha)
    df_event_price=pd.read_pickle(datafile)

    df_event_price=df_event_price[["Date","event","Price"]]
    X=np.array(df_event_price.event.to_numpy().reshape(-1).tolist())
    y=df_event_price.Price.to_numpy().reshape(-1)


    ts_cv=get_TS_cv()
    random_cv=get_random_cv()

    svr_ts_result=evaluate(model=SVR(),X=X,y=y,cv=ts_cv)
    svr_ran_result=evaluate(model=SVR(),X=X,y=y,cv=random_cv)

    lasso_TS_result=evaluate(model=Lasso(alpha=alpha),X=X,y=y,cv=ts_cv)
    lasso_ran_result=evaluate(model=Lasso(alpha=alpha),X=X,y=y,cv=random_cv)


