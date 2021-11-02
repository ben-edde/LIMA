import pandas as pd


def prepare_X(file_name: str, feature_name: str, lag: int):
    """
    read feature data file and prepare independent variables of [t-lag, t-1], where len(df)=t+3.

    Args:
        file_name (str): path to file + file name
        feature_name (str): column name of feature
        lag (int): order to shift; >= 1 
    """
    df = pd.read_csv(file_name)
    df.Date = pd.to_datetime(df.Date)
    base_shift = -3
    for i in range(1, lag + 1):
        to_shift = base_shift - i
        df[f"{feature_name}_t-{i}"] = df.shift(to_shift)[feature_name]
    df.dropna(inplace=True)
    df.drop(feature_name, inplace=True, axis=1)
    df.to_csv(f"X_{feature_name}_t-1_t-{lag}.csv", index=False)
    return df
