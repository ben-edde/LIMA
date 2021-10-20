# Import libraries
import pandas as pd
import pickle
import numpy as np
import time
from openie import StanfordOpenIE
from gensim.models import Word2Vec

df_events = pd.read_pickle("event_tuple_w2v.pkl")
df_price= pd.read_csv("Cushing_OK_WTI_Spot_Price_FOB.csv")


# find daily event representation
df_events["event_sum"]=df_events.subject+df_events.relation+df_events.object
df_daily_event=pd.DataFrame(columns=["Date","Event"])
for date, group in df_events.groupby(["Date"]):
    daily_mean=np.mean(group["event_sum"].to_numpy())
    df_daily_event=df_daily_event.append({"Date":date,"Event":daily_mean},ignore_index=True)


# find common keys
df_price.Date=pd.to_datetime(df_price.Date)
df_price.set_index("Date",inplace=True)
df_daily_event.Date=pd.to_datetime(df_daily_event.Date)
df_daily_event.set_index("Date",inplace=True)
good_keys = df_price.index.intersection(df_daily_event.index)
df_price=df_price.loc[good_keys]
df_daily_event=df_daily_event.loc[good_keys]

# export if needed