import pandas as pd
import numpy as np
import matplotlib.pyplot
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

bucket = "default_bucket"
measurement = "brent"

def main():
    price_oil = pd.read_csv("data/BrentOilPrices.csv", dtype={"Price": float})
    price_oil.set_index("Date", inplace=True)
    price_oil.index = pd.to_datetime(price_oil.index)

    #client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    client=influxdb_client.InfluxDBClient.from_config_file("config.ini")
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket,
                    record=price_oil,
                    data_frame_measurement_name=measurement)

    client.close()

if __name__ == "__main__":
    main()