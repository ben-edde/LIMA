import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sacred import Experiment
from sacred.observers import FileStorageObserver

exp = Experiment('WTI_RedditNews_sentiment-polarity_LASSO_SVM')
exp.observers.append(FileStorageObserver('runs'))


@exp.capture
def find_sentiment(df_news):
    df_result = df_news[["Date"]]
    df_result["Polarity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.polarity, axis=1)
    df_daily_averaged_sentiment_score = df_result.groupby(['Date']).mean()
    return df_daily_averaged_sentiment_score


@exp.capture
def prepare_X(df, feature_name: str, lag: int):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    base_shift = -3
    for i in range(1, lag + 1):
        to_shift = base_shift - i
        print(f"{feature_name}| lag: {lag} | to_shift: {to_shift}")
        df[f"{feature_name}_t-{i}"] = df.shift(to_shift)[feature_name]
    df.dropna(inplace=True)
    df.drop(feature_name, inplace=True, axis=1)
    return df


@exp.capture
def prepare_y(df: str):
    df.Date = pd.to_datetime(df.Date)
    df["Price_t+3"] = df.shift(0)["Price"]
    df["Price_t+2"] = df.shift(-1)["Price"]
    df["Price_t+1"] = df.shift(-2)["Price"]
    df["Price_t"] = df.shift(-3)["Price"]
    df.dropna(inplace=True)
    df.drop("Price", inplace=True, axis=1)
    df.set_index("Date", inplace=True)
    return df


@exp.capture
def one_fold(model, X, y, train_ratio):
    train_size = int(len(X) * float(train_ratio))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"""Test error
    {model.__class__.__name__}:
    score: {score}
    """)
    return score


@exp.capture
def ten_fold(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(model,
                            X,
                            y,
                            scoring=[
                                'neg_mean_absolute_error',
                                'neg_root_mean_squared_error',
                                'neg_mean_absolute_percentage_error'
                            ],
                            cv=cv,
                            n_jobs=-1)
    print(f"""Test error (cross-validated performance)
    {model.__class__.__name__}:
    MAE = {-scores["test_neg_mean_absolute_error"].mean():.3f}
    RMSE = {-scores["test_neg_root_mean_squared_error"].mean():.3f}
    MAPE = {-scores["test_neg_mean_absolute_percentage_error"].mean():.3f}
    """)
    return scores


@exp.automain
def main():
    df_news = pd.read_csv("RedditNews_filtered.csv")
    df_senti = find_sentiment(df_news)
    df_X = prepare_X(df_senti, "Polarity", 5)

    df_price = pd.read_csv("WTI_Spot.csv")
    df_price.Date = pd.to_datetime(df_price.Date)
    df_y = prepare_y(df_price)
    df_Xy = df_X.merge(df_y, left_index=True, right_index=True)

    df_Xy_0 = df_Xy.drop(["Price_t+3", "Price_t+2", "Price_t+1"], axis=1)

    X = df_Xy_0.to_numpy()[:, :-1]
    y = df_Xy_0.to_numpy()[:, -1]

    model = Lasso(alpha=1.0)
    result = one_fold(model, X, y, 0.7)
    exp.log_scalar("LASSO one-fold: ", result)
    result = ten_fold(model, X, y)
    [
        exp.log_scalar(f"{model.__class__.__name__} ten-fold({each})",
                       result[each].mean()) for each in result
    ]
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    result = one_fold(model, X, y, 0.7)
    exp.log_scalar("SVR one-fold: ", result)
    result = ten_fold(model, X, y)
    [
        exp.log_scalar(f"{model.__class__.__name__} ten-fold({each})",
                       result[each].mean()) for each in result
    ]
