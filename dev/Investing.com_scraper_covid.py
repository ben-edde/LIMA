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
    try:
        # latest news
        if 'ago' in time_str:
            time_str=time_str[3:]
            amount=int(time_str.split(' ')[0])
            unit=time_str.split(' ')[1]
            if unit in ["hour","hours"]:
                return (datetime.datetime.utcnow() -
                                datetime.timedelta(hours=amount)).date()
            elif unit in ["minute","minutes"]:
                return (datetime.datetime.utcnow() -
                                datetime.timedelta(minutes=amount)).date()
            elif unit in ["second","seconds"]:
                return (datetime.datetime.utcnow() -
                                datetime.timedelta(seconds=amount)).date()
        else:
            return datetime.datetime.strptime(time_str, " - %b %d, %Y").date()
    except Exception as e:
        logging.exception(f"convert_time_str_to_dt: {e}")
        return datetime.datetime.utcnow().date()


def main():
    logging.info(f"Investing.com scraper started")
    try:
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        mongo_db = mongo_client["lima"]
        mongo_collection = mongo_db["investing_news_covid"]
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        driver = webdriver.Firefox(
            executable_path=f"{HOME}/dev/drivers/geckodriver", options=opts)
        # file_name = "Investing.com_commodities-news.csv"
        # file_destination = f"{HOME}/data/fresh/{file_name}"
        for p in range(1, 652):
            for retry in range(5):
                try:
                    driver.get(
                        f"https://www.investing.com/news/coronavirus/{p}")
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
                    if len(df)==0:
                        continue
                    df.columns = ["Date", "ID", "News"]
                    df.Date = df.Date.map(convert_time_str_to_dt)
                    # df.to_csv(file_destination,
                    #           index=False,
                    #           mode="a+",
                    #           header=False)
                    df.Date = pd.to_datetime(df.Date)
                    for each in df.to_dict(orient='records'):
                        try:
                            mongo_collection.insert_one(each)
                        except Exception as e:
                            if e.details["code"] == 11000:
                                pass
                            else:
                                logging.exception(
                                    f"Investing.com forex scraper(p= {p}; retry={retry}) writing to MongoDB: {e}"
                                )
                    break
                except Exception as e:
                    logging.exception(f"Investing.com forex scraper(p= {p}; retry={retry}): {e}")
                    logging.exception(f"results: {results}")
                    time.sleep(5)
        driver.close()
        logging.info("Investing.com forex scraper finished job.")
    except Exception as e:
        logging.exception(f"Investing.com forex scraper: {e}")
        time.sleep(5)


if __name__ == "__main__":
    main()