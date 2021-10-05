from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


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
