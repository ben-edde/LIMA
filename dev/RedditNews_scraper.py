import datetime
import logging
import os
import numpy as np
import pandas as pd
import requests
import time

logging.basicConfig(
    filename=os.environ["PublicSeaLogPath"],
    filemode='a',
    format=
    '%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(message)s',
    level=logging.INFO)
HOME = os.environ['LIMA_HOME']


def main():
    for t in range(5):
        try:
            logging.info(f"RedditNews scraper running on ({t+1}) time")
            url = "https://www.reddit.com/r/worldnews/.json"
            response = requests.get(url,headers = {'User-agent': 'BOT'})
            response_json = response.json()
            if "data" not in response_json.keys():
                logging.error("no data")
                logging.info(f"response_json keys: {response_json.keys()}")
                logging.info(f"msg: {response_json}")
                continue
            # num_item=response_json['data']['dist']
            results = [(each['data']['score'], each['data']['title'])
                       for each in response_json['data']['children']]

            df = pd.DataFrame(results)
            df.columns = ['score', 'News']
            df = df.sort_values("score", ascending=False, ignore_index=True)
            df = df.drop("score", axis=1)
            df["Date"] = datetime.datetime.utcnow().date()
            df = df[["Date", "News"]]
            file_name = "RedditNews.csv"
            file_destination = f"{HOME}/data/fresh/{file_name}"
            df.to_csv(file_destination, index=False, mode="a+", header=False)
            logging.info("RedditNews scraper saved file.")
            break
        except Exception as e:
            logging.exception(e)
            time.sleep(10)


if __name__ == "__main__":
    main()