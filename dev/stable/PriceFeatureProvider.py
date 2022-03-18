import os

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient

from FeatureProvider import FeatureProvider

HOME = os.environ['LIMA_HOME']


class PriceFeatureProvider(FeatureProvider):
    def __init__(self) -> None:
        pass

    def get_raw_data(self, mode):
        if mode == "forecast":
            since = "-50d"
        else:
            since = "2011-03-01"
        client = InfluxDBClient.from_config_file(
            f"{HOME}/dev/DB/influxdb_config.ini")
        query_api = client.query_api()
        df_WTI = query_api.query_data_frame(f"""
        from(bucket: "dummy")
        |> range(start:{since}, stop: now())
        |> filter(fn: (r) => r["_measurement"] == "WTI") 
        |> filter(fn: (r) => r["type"] == "closing_price") 
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> drop(columns: ["_start", "_stop"])
        """)
        df_WTI = df_WTI[["_time", "CLC4", "CLC3", "CLC2", "CLC1"]]
        df_WTI.columns = ["Date", "CLC4", "CLC3", "CLC2", "CLC1"]
        df_WTI.set_index("Date", inplace=True)
        df_WTI.index = df_WTI.index.map(lambda each: each.date())
        df_WTI.index = pd.to_datetime(df_WTI.index)
        return df_WTI

    def get_time_feature(self, df):
        month = [each.month for each in df.index]
        day = [each.day for each in df.index]
        day_in_week = [each.weekday() for each in df.index]
        df_dt = pd.DataFrame()
        df_dt["month"] = month
        df_dt["day"] = day
        df_dt["day_in_week"] = day_in_week
        df_dt.index = df.index
        return df_dt

    def get_feature(self, mode):
        df_WTI = self.get_raw_data(mode).dropna()
        df_dt = self.get_time_feature(df_WTI)
        return df_WTI, df_dt
