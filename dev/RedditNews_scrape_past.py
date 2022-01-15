import requests
import datetime
import pandas as pd
all_results=[]

nums=3
for t in range(1,nums):
    url=f"https://api.pushshift.io/reddit/search/submission/?subreddit=worldnews&sort_type=score&before={t+1}d&after={t+2}d&fields=title&size=25"
    response=requests.get(url)
    response_json=response.json()
    results=[(each['title']) for each in response_json['data']]
    df=pd.DataFrame(results)
    df["Date"]=datetime.date.today()-datetime.timedelta(days=t)
    all_results.append(df)
df_news=pd.concat(all_results,ignore_index=True)
df_news.columns=["News","Date"]
df_news.to_csv(index=False)