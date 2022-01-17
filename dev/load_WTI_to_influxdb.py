import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS,ASYNCHRONOUS,WriteOptions
import numpy as np
import pandas as pd
import os
HOME = os.environ["LIMA_HOME"]

client= InfluxDBClient.from_config_file(f"{HOME}/dev/DB/influxdb_config.ini")
write_api=client.write_api(write_options=WriteOptions(batch_size=5_000, flush_interval=1_000))
df_WTI= pd.read_csv(f"{HOME}/data/fresh/WTI_EIA.csv")
df_WTI.Date=df_WTI.Date.map(lambda x: datetime.datetime.combine(datetime.datetime.strptime(x,"%Y-%m-%d"),datetime.time(17,0)))
df_WTI.Date=pd.to_datetime(df_WTI.Date)
df_WTI.Date=df_WTI.Date.dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
df_WTI.set_index("Date",inplace=True)
df_WTI["type"]="closing_price"
write_api.write(bucket="dummy",record=df_WTI, data_frame_measurement_name="WTI",data_frame_tag_columns=["type"])
write_api.close()
client.close()