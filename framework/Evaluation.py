"""
Evaluation functions and logging config. To be copied to experiment file instead of being imported to avoid import problem.
"""
import logging
import os
from sklearn.model_selection import cross_validate, TimeSeriesSplit

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format='%(asctime)s %(filename)s %(funcName)s %(levelname)s %(message)s',
    level=logging.DEBUG)


def get_TS_cv(k=10, horizon=0):
    return TimeSeriesSplit(
        n_splits=k,
        gap=horizon - 1,
        test_size=1,
    )


# does not make sense to future data to predict past: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
# def get_random_cv(k=10):
#     return RepeatedKFold(n_splits=k, random_state=42)


def evaluate(model, X, y, cv):
    try:
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
        Forecast Error ({k}-fold cross-validation; horizon={cv.gap+1})
        {model.__class__.__name__}:
        MAE = {mae.mean():.3f} +/- {mae.std():.3f}
        RMSE = {rmse.mean():.3f} +/- {rmse.std():.3f}
        MAPE = {mape.mean():.3f} +/- {mape.std():.3f}
        """
        print(msg)
        logging.info(msg)
        return cv_results
    except Exception as e:
        logging.exception("EXCEPTION: %s", e, exc_info=True)
