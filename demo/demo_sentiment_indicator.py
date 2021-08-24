import pandas as pd
from textblob import TextBlob
import math

df_news = pd.read_csv("data/Bai_news_headlines.csv")
df_news.date = pd.to_datetime(df_news.date)

polarity_list = []
for each in df_news['headlines']:
    testimonial = TextBlob(each)
    polarity_list.append(testimonial.sentiment.polarity)

df_news['polarity'] = polarity_list
# take daily mean
df_daily_averaged_sentiment_score = df_news.groupby(['date']).mean()
df_daily_averaged_sentiment_score.columns = ['daily_polarity']

cumulated_sentiment_polarity_list = []
# iterate through all days' sentiment scores [0,n-1]
for current_day_num in range(len(df_daily_averaged_sentiment_score)):
    daily_news_sentiment = df_daily_averaged_sentiment_score.iloc[
        current_day_num].daily_polarity
    # for current day (t-th day), back track from [0,t-1]
    for back_track_day_num in range(current_day_num):
        # take exponential decay for polarity on i-th day where i:[0,t-1]
        daily_news_sentiment += df_daily_averaged_sentiment_score.iloc[
            back_track_day_num].daily_polarity * math.exp(
                -(current_day_num - back_track_day_num) / 7)
    cumulated_sentiment_polarity_list.append(daily_news_sentiment)

df_daily_averaged_sentiment_score[
    'cumulated_polarity'] = cumulated_sentiment_polarity_list

# date became index after groupby
df_daily_averaged_sentiment_score.to_csv("sentiment_indicators.csv",
                                         index=True)
