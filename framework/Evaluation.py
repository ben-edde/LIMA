from sklearn.model_selection import cross_validate, RepeatedKFold, TimeSeriesSplit


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
    print(f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
          f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}")
    k = cv.get_n_splits()
    print(f"""
    Forecast Error ({k}-fold cross-validated performance):
    {model.__class__.__name__}:
    MAE = {mae.mean():.3f}
    RMSE = {rmse.mean():.3f}
    MAPE = {mape.mean():.3f}
    """)
    return cv_results
