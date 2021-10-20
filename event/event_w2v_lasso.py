import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score,cross_validate 
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

# event tuple transformed with w2v
df_events=pd.read_pickle("event_tuple_w2v.pkl")

# find daily event representation
df_events["event_sum"]=df_events.subject+df_events.relation+df_events.object
df_daily_event=pd.DataFrame(columns=["Date","Event"])
for date, group in df_events.groupby(["Date"]):
    daily_mean=np.mean(group["event_sum"].to_numpy())
    df_daily_event=df_daily_event.append({"Date":date,"Event":daily_mean},ignore_index=True)

# read price data
df_price=pd.read_csv("Cushing_OK_WTI_Spot_Price_FOB.csv")


# set date as index for both df
df_price.Date=pd.to_datetime(df_price.Date)
df_price.set_index("Date",inplace=True)
df_daily_event.Date=pd.to_datetime(df_daily_event.Date)
df_daily_event.set_index("Date",inplace=True)

# merge to ensure consistent shape
df_Xy=df_daily_event.merge(df_price,left_index=True,right_index=True)

# X must be flatten here to be in shape of (xxx,yyy) instead of (xxx,). Otherwise, model cannot read X
X, y =np.array(df_Xy["Event"].to_numpy().reshape(-1).tolist()), df_Xy["Price"].to_numpy().reshape(-1)

train_ratio=int(len(X)*0.7)
X_train,X_test =X[:train_ratio],X[train_ratio:]
y_train,y_test =y[:train_ratio],y[train_ratio:]

model = Lasso(alpha=1.0)
model.fit(X_train,y_train)
model.score(X_test,y_test)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_validate(model, X, y, scoring=['neg_mean_absolute_error','neg_root_mean_squared_error','neg_mean_absolute_percentage_error'], cv=cv, n_jobs=-1)

print(
    f"""Test error (cross-validated performance)
    {model.__class__.__name__}:
    MAE = {-scores["test_neg_mean_absolute_error"].mean():.3f}
    RMSE = {-scores["test_neg_root_mean_squared_error"].mean():.3f}
    MAPE = {-scores["test_neg_mean_absolute_percentage_error"].mean():.3f}
    """)

