import pandas as pd
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.statespace.sarimax import SARIMAX
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


def write_InfluxDB(url, token, org, bucket, measurement, df):
    client = influxdb_client.InfluxDBClient.from_config_file("config.ini")
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket, record=df, data_frame_measurement_name=measurement)


def main():
    price_oil = pd.read_csv("data/BrentOilPrices.csv", dtype={"Price": float})
    price_oil.set_index("Date", inplace=True)
    price_oil.index = pd.to_datetime(price_oil.index)
    # price_oil['1difference'] = price_oil['Price'] - price_oil['Price'].shift(1)
    # price_oil['2difference'] = price_oil['1difference'] - price_oil[
    #     '1difference'].shift(1)
    # price_oil['Seasonal_Difference'] = price_oil['Price'] - price_oil[
    #     'Price'].shift(12)
    model = SARIMAX(price_oil['Price'],
                    order=(1, 2, 1),
                    seasonal_order=(1, 0, 0, 12))
    result = model.fit()

    new_dates = [price_oil.index[-1] + DateOffset(days=x) for x in range(1, 4)]
    pred = pd.DataFrame(
        result.predict(start=price_oil.shape[0] + 1,
                       end=price_oil.shape[0] + len(new_dates)))
    pred['date'] = new_dates
    pred.set_index('date', inplace=True)
    pred.columns = ['pred']
    #write_InfluxDB(pred)


if __name__ == "__main__":
    main()