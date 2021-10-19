"""
Both news and prices data would be used. It is always more news date than price date (due to holiday). Filter news with respect to price.
"""
import pandas as pd
df_news = pd.read_csv("RedditNews.csv")
df_price=pd.read_csv("Cushing_OK_WTI_Spot_Price_FOB.csv")
df_price.Date=pd.to_datetime(df_price.Date)
df_price.set_index("Date",inplace=True)

df_filtered_news=df_news[df_news.Date.isin(df_price.Date)]

df_filtered_news.to_csv("RedditNews_filtered.csv",index=False)