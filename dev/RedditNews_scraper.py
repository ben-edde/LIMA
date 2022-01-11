import datetime
import logging
import os
import numpy as np
import pandas as pd
import requests

HOME = os.environ['LIMA_HOME']

def main():
    url="https://www.reddit.com/r/worldnews/.json"
    response=requests.get(url)
    response_json=response.json()

    # num_item=response_json['data']['dist']
    results=[(each['data']['score'],each['data']['title']) for each in response_json['data']['children']]

    df=pd.DataFrame(results)
    df.columns=['score','News']
    df=df.sort_values("score",ascending=False,ignore_index=True)
    df=df.drop("score",axis=1)
    df["Date"]=datetime.date.today()
    df=df[["Date","News"]]
    file_name=datetime.date.today().strftime("RedditNews_%Y-%m-%d.csv")
    file_destination=f"{HOME}/data/fresh/{file_name}"
    df.to_csv(file_destination,index=False)

if __name__=="__main__":
    main()