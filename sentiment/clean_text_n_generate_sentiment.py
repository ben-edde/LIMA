import os
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from textblob import TextBlob

HOME = os.environ['LIMA_HOME']


def clean(text: str) -> list:
    """
    clean text with tokenization; stemming; removing stop word, punctuation, number, and empty string.

    Args:
        text (str): text

    Returns:
        list: cleaned text as list of tokenized str
    """

    # to list of token
    text = word_tokenize(text)

    # stemming and convert to lower case if not proper noun: punctuation and stop word seem to help POS tagging, remove them after stemming
    word_tag = pos_tag(text)
    porter = PorterStemmer()
    text = [
        porter.stem(each[0])
        if each[1] != "NNP" and each[1] != "NNPS" else each[0]
        for each in word_tag
    ]

    # remove stop word: it seems stemming skip stop word; OK to remove stop word after stemming;
    stop_word = set(stopwords.words('english'))
    text = [each for each in text if not each in stop_word]

    # remove punctuation
    text = [
        each.translate(str.maketrans('', '', string.punctuation))
        for each in text
    ]
    # text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text) # if using re

    # convert number to <NUM>
    text = ["<NUM>" if each.isdigit() else each for each in text]

    # remove empty string
    text = [each for each in text if each != ""]

    return text


def get_sentiment(df):
    df["Polarity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.polarity, axis=1)
    df["Subjectivity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.subjectivity, axis=1)
    return df


def get_sentiment_aggregated(df):
    df["Polarity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.polarity, axis=1)
    df["Subjectivity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.subjectivity, axis=1)
    df_daily_averaged_sentiment_score = df.groupby(['Date']).mean()
    return df_daily_averaged_sentiment_score


df_news = pd.read_csv(
    f"{HOME}/data/reuse/RedditNews_WTI/RedditNews_2008-06-09_2016-07-01.csv")
df_price = pd.read_csv(
    f"{HOME}/data/reuse/RedditNews_WTI/WTI_Spot_2008-06-09_2016-07-01.csv")
df_news.News = df_news.News.apply(lambda r: " ".join(clean(r)))

df_news_price = df_news.merge(df_price, on="Date", how="outer")

df_sentiment_aggregated = get_sentiment_aggregated(df_news_price)
df_sentiment_aggregated.reset_index(inplace=True)
df_sentiment = get_sentiment(df_news_price)

df_sentiment_aggregated = df_sentiment_aggregated[[
    "Date", "Polarity", "Subjectivity", "Price"
]]
df_sentiment_aggregated.to_pickle(
    "WTI_Spot_n_RedditNews_2008-06-09_2016-07-01_sentiment_aggregated.pkl")

df_sentiment = df_sentiment[["Date", "Polarity", "Subjectivity", "Price"]]
df_sentiment.to_pickle(
    "WTI_Spot_n_RedditNews_2008-06-09_2016-07-01_sentiment.pkl")
