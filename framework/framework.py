import pandas as pd
import datetime
from configparser import ConfigParser
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def export_result(df: pd.DataFrame, name: str):
    ts = int(datetime.datetime.now().timestamp())
    df.to_pickle(f"{name}_{ts}.pkl")


def load_config(cfg_filename: str = "cfg.ini") -> ConfigParser:
    config = ConfigParser()
    config.read_file(open(cfg_filename))
    return config


def update_config(config: ConfigParser, cfg_filename: str = "cfg.ini"):
    with open(cfg_filename, 'w') as configfile:
        config.write(configfile)


def load_data(filename, dataname, ratio=1, csv_file=True) -> pd.DataFrame:
    if csv_file:
        df = pd.read_csv(filename)
        df.columns = ["date", dataname]
        df.date = pd.to_datetime(df.date)
        if ratio < 1:
            df = df.iloc[:int(len(df) * ratio)]
        return df
    else:
        return pd.read_pkl(filename)


def split_train_test_data(df_xy):
    training_size = int(len(df_xy) * 2 / 3)
    train_x = df_xy.iloc[:training_size, :-1].values
    train_y = df_xy.iloc[:training_size, -1].values
    test_x = df_xy.iloc[training_size:, :-1].values
    test_y = df_xy.iloc[training_size:, -1].values
    return train_x, train_y, test_x, test_y


def evaluate(y_true, y_pred):
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return pd.DataFrame({"rmse": [rmse], "mae": [mae], "mape": [mape]})


def record(df, description):
    ts = int(datetime.datetime.now().timestamp())
    df["description"] = description
    df.to_csv(f"EvaluationResult_{ts}.csv", index=False)


def run_experiment(ratio, feature_method, model_method, description):
    config = load_config(cfg_filename="setup.cfg")
    # change_in_cleaning = config["cleaning"].getboolean("change")
    # change_in_extract = config["extract"].getboolean("change")
    # change_in_select = config["select"].getboolean("change")
    # change_in_model = config["model"].getboolean("change")
    news_filename = config["data"]["news_data_file"]
    price_filename = config["data"]["price_data_file"]
    df_news = load_data(news_filename, "news", ratio)
    df_prices = load_data(price_filename, "price")
    df_prices.set_index("date", inplace=True)
    df_features = feature_method(df_news)
    df_xy = df_features.merge(df_prices,
                              how="inner",
                              left_index=True,
                              right_index=True)
    train_x, train_y, test_x, test_y = split_train_test_data(df_xy)
    model = model_method()
    print(train_x.shape, train_y.shape)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    df_evaluation_result = evaluate(test_y, pred_y)
    record(df_evaluation_result, description)
