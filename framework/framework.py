import pandas as pd
import datetime
from configparser import ConfigParser
from typing import Tuple


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


def load_data(filename, csv_file=True) -> pd.DataFrame:
    if csv_file:
        return pd.read_csv(filename)
    else:
        return pd.read_pkl(filename)


def cleaning(df_news: pd.DataFrame,
             df_price: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df_news, df_price


def extract_feature():
    pass


def select_feature():
    pass


def split_dataset():
    pass


def train_model():
    pass


def evaluation():
    pass


def record():
    pass


def run():
    config = load_config()
    change_in_cleaning = config["cleaning"].getboolean("change")
    change_in_extract = config["extract"].getboolean("change")
    change_in_select = config["select"].getboolean("change")
    change_in_model = config["model"].getboolean("change")
    news_filename = config["data"]["news_data_file"]
    price_filename = config["data"]["price_data_file"]
    df_news = load_data(news_filename)
    df_price = load_data(price_filename)
    if change_in_cleaning:
        df_news, df_price = cleaning(df_news, df_price)
    else:
        df_news = load_data(config["cleaning"]["latest_news_result"],
                            csv_file=False)
        df_price = load_data(config["cleaning"]["latest_price_result"],
                             csv_file=False)
    # extract_feature()
    # select_feature()
    # split_dataset()
    # train_model()
    # evaluation()
    # record()


def main():
    pass


if __name__ == "__main__":
    main()