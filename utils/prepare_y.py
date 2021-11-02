import pandas as pd


def prepare_y(price_file_name: str):
    """
    read price data file and prepare dependent variables of [t, t+3], where len(df)=t+3.


    Args:
        price_file_name (str): path to file + file name
    """
    df = pd.read_csv(price_file_name)
    df.Date = pd.to_datetime(df.Date)
    df["Price_t+3"] = df.shift(0)["Price"]
    df["Price_t+2"] = df.shift(-1)["Price"]
    df["Price_t+1"] = df.shift(-2)["Price"]
    df["Price_t"] = df.shift(-3)["Price"]
    df.dropna(inplace=True)
    df.drop("Price", inplace=True, axis=1)
    df.to_csv("WTI_Spot_y[t-t+3].csv", index=False)
    return df
