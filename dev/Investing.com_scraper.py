import datetime
import logging
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
import datetime
import time

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format=
    '%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(message)s',
    level=logging.INFO)
HOME = os.environ['LIMA_HOME']


def main():
    logging.info(f"Investing.com scraper started")
    try:
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        driver = webdriver.Firefox(
            executable_path=f"{HOME}/dev/drivers/geckodriver", options=opts)
        file_name = "Investing.com_commodities-news(Debug).csv"
        file_destination = f"{HOME}/data/fresh/{file_name}"
        for p in range(2, 2187):
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
                    df.Date = df.Date.map(
                        lambda each: datetime.datetime.strptime(
                            each, " - %b %d, %Y").date())
                    # df.to_csv(file_destination,
                    #           index=False,
                    #           mode="a+",
                    #           header=False)
                    break
                except Exception as e:
                    logging.exception(f"Investing.com scraper(p= {p}): {e}")
                    print(f"Investing.com scraper(p= {p}): {e}")
                    time.sleep(5)
        driver.close()
        logging.info("Investing.com scraper finished job.")
    except Exception as e:
        logging.exception(f"Investing.com scraper: {e}")
        time.sleep(5)


if __name__ == "__main__":
    main()