"""
To be executed at 6pm EST where closing price of that day will be available.
"""
import datetime
import logging
import os
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
import datetime
import time
from pytz import timezone
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format=
    "%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(message)s",
    level=logging.INFO)
HOME = os.environ["LIMA_HOME"]


def main():
    # follow date of exchange
    now_EST = datetime.datetime.now(timezone("US/Eastern"))
    if now_EST.weekday() in [4, 5]:
        print(f"WTI scraper start on non-trading day: {now_EST.weekday()}")
        return
    for t in range(5):
        logging.info(f"WTI scraper running on ({t+1}) time")
        try:
            opts = FirefoxOptions()
            opts.add_argument("--headless")
            driver = webdriver.Firefox(
                executable_path=f"{HOME}/dev/drivers/geckodriver",
                options=opts)
            all_results = []
            for i in range(1, 5):
                try:
                    label = f"CLC{i}"
                    driver.get(f"https://www.cnbc.com/quotes/{label}")
                    summary_label = driver.find_elements(
                        By.CLASS_NAME, "Summary-label")
                    summary_value = driver.find_elements(
                        By.CLASS_NAME, "Summary-value")
                    results = [(each[0].text, each[1].text)
                               for each in zip(summary_label, summary_value)]
                    df = pd.DataFrame(results)
                    df.columns = ["Type", "Data"]
                    closing_price = df[df.Type ==
                                       "Prev Close"].Data.to_numpy()[0]
                    all_results.append((label, closing_price))
                except Exception as e:
                    logging.exception(f"WTI scraper on ({t+1}) time: {e}")
                    time.sleep(5)
                    driver = webdriver.Firefox(
                        executable_path=f"{HOME}/dev/drivers/geckodriver",
                        options=opts)
            df = pd.DataFrame(all_results)
            df.columns = ["Label", "Price"]
            # on Sunday, the last trading day is Friday
            if now_EST.weekday() == 6:
                df["Date"] = (now_EST - datetime.timedelta(days=2)).date()
            else:
                df["Date"] = now_EST.date()
            df_p = df.pivot(index="Date", columns="Label", values="Price")
            file_name = "WTI_4C_CNBC.csv"
            file_destination = f"{HOME}/data/fresh/{file_name}"
            df_p.to_csv(file_destination, mode="a+", header=False)
            driver.close()
            logging.info("WTI scraper saved file.")

            # write to InfluxDB
            try:
                # EST 5pm(market close) == UTC 10pm
                df_p.index = df_p.index.map(lambda x: datetime.datetime.
                                            combine(x, datetime.time(22, 0)))
                df_p = df_p.astype("double")
                df_p["type"] = "closing_price"
                client = InfluxDBClient.from_config_file(
                    f"{HOME}/dev/DB/influxdb_config.ini")
                write_api = client.write_api(write_options=SYNCHRONOUS)
                write_api.write(bucket="dummy",
                                record=df_p,
                                data_frame_measurement_name="WTI",
                                data_frame_tag_columns=["type"])
                client.close()
            except Exception as e:
                logging.exception(f"WTI scraper writing to InfluxDB: {e}")
            break
        except Exception as e:
            logging.exception(f"WTI scraper on ({t+1}) time: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()