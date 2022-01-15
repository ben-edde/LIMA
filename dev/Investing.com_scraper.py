import datetime
import logging
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
import datetime
import time
import pymongo

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format=
    '%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(message)s',
    level=logging.INFO)
HOME = os.environ['LIMA_HOME']


def convert_time_str_to_dt(time_str):
    # latest news
    if 'ago' in time_str:
        hours_to_shift = datetime.datetime.strptime(time_str,
                                                    ' - %H hours ago').hour
        return (datetime.datetime.utcnow() -
                datetime.timedelta(hours=hours_to_shift)).date()
    else:
        return datetime.datetime.strptime(time_str, " - %b %d, %Y").date()


def main():
    logging.info(f"Investing.com scraper started")
    try:
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        mongo_db = mongo_client["lima"]
        mongo_collection = mongo_db["investing_news"]
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        driver = webdriver.Firefox(
            executable_path=f"{HOME}/dev/drivers/geckodriver", options=opts)
        file_name = "Investing.com_commodities-news.csv"
        file_destination = f"{HOME}/data/fresh/{file_name}"
        for p in range(1, 3):
            for retry in range(5):
                try:
                    driver.get(
                        f"https://www.investing.com/news/commodities-news/{p}")
                    title_text = driver.find_elements(
                        By.XPATH,
                        "//div[@class='largeTitle']/article//a[@class='title']"
                    )
                    date_text = driver.find_elements(
                        By.XPATH,
                        "//div[@class='largeTitle']/article//span[@class='date']"
                    )
                    articles = driver.find_elements(
                        By.XPATH,
                        "//div[@class='largeTitle']/article[@class='js-article-item articleItem   ']"
                    )
                    results = [(each[0].text, each[1].get_attribute("data-id"),
                                each[2].text)
                               for each in zip(date_text, articles, title_text)
                               ]
                    df = pd.DataFrame(results)
                    df.columns = ["Date", "ID", "News"]
                    df.Date = df.Date.map(convert_time_str_to_dt)
                    df.to_csv(file_destination,
                              index=False,
                              mode="a+",
                              header=False)
                    df.Date = pd.to_datetime(df.Date)
                    for each in df.to_dict(orient='records'):
                        try:
                            mongo_collection.insert_one(each)
                        except Exception as e:
                            if e.details["code"] == 11000:
                                pass
                            else:
                                logging.exception(
                                    f"Investing.com scraper(p= {p}; retry={retry}) writing to MongoDB: {e}"
                                )
                    break
                except Exception as e:
                    logging.exception(f"Investing.com scraper(p= {p}): {e}")
                    print(f"Investing.com scraper(p= {p}; retry={retry}): {e}")
                    time.sleep(5)
        driver.close()
        logging.info("Investing.com scraper finished job.")
    except Exception as e:
        logging.exception(f"Investing.com scraper: {e}")
        time.sleep(5)


if __name__ == "__main__":
    main()