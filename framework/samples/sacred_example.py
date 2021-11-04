import pandas as pd
import numpy as np
import datetime
from configparser import ConfigParser
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sacred import Experiment
from sacred.observers import FileStorageObserver
ex = Experiment('testing_01')
ex.observers.append(FileStorageObserver('runs'))

def evaluate(y_true, y_pred):
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    #return pd.DataFrame({"rmse":[rmse], "mae":[mae],"mape":[mape]})
    return {"rmse":rmse, "mae":mae,"mape":mape}

def predict():
    return [1,2,3,4,5]


@ex.automain
def run(num):
    pred=np.array(predict())
    ans=np.array([1,2,3,4,4])
    ex.log_scalar("evaluation", evaluate(ans,pred))
    ans=np.array([1,2,3,4,5])
    ex.log_scalar("evaluation", evaluate(ans,pred))
    print(
        f"""
        num={num}
        """)
    return 
