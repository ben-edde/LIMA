import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


def write_InfluxDB(df, bucket='default_bucket', measurement='brent'):
    client = influxdb_client.InfluxDBClient.from_config_file("config.ini")
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket, record=df, data_frame_measurement_name=measurement)


# read original oil price which defined scale
df_price = pd.read_csv("data/Bai_origin_oil.csv")
df_price.date = pd.to_datetime(df_price.date)
mm = MinMaxScaler()
mm.fit(np.array(df_price['price']).reshape(-1, 1))

# read normalized predicted price
df_normalized_old_n_new = pd.read_csv("normalized_polarity_price_h1_h2_h3.csv")
df_normalized_old_n_new.date = pd.to_datetime(df_normalized_old_n_new.date)
df_normalized_old_n_new.set_index('date', inplace=True)
df_pred = df_normalized_old_n_new[["h1", "h2", "h3"]].iloc[-3:]

# invert transform
df_pred.h1 = mm.inverse_transform(np.array(df_pred.h1).reshape(-1, 1))
df_pred.h2 = mm.inverse_transform(np.array(df_pred.h2).reshape(-1, 1))
df_pred.h3 = mm.inverse_transform(np.array(df_pred.h3).reshape(-1, 1))

# write original oil price (if it was not written yet)
# df_price.set_index('date', inplace=True)
# write_InfluxDB(df_Bai_origin_oil, measurement='WTI')

# write predicted price
write_InfluxDB(df_pred, measurement='WTI')
