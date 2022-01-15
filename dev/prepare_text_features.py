import os
import string
import fasttext
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from textblob import TextBlob
import joblib

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


df_news = pd.read_csv(
    f"{HOME}/data/reuse/RedditNews_WTI/RedditNews_2008-06-09_2016-07-01.csv")
df_news.News = df_news.News.apply(lambda r: " ".join(clean(r)))
lda_model = joblib.load(f"{HOME}/dev/models/LDA_5_RedditNews_fasttext.joblib")
ft_scaler = joblib.load(
    f"{HOME}/dev/models/MinMaxScaler_RedditNews_fasttext.joblib")
ft_model = fasttext.load_model(f"{HOME}/data/big/cc.en.300.bin")


def get_sentiment_aggregated(df):
    df["Polarity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.polarity, axis=1)
    df["Subjectivity"] = df_news.apply(
        lambda row: TextBlob(row['News']).sentiment.subjectivity, axis=1)
    df_daily_averaged_sentiment_score = df.groupby(['Date']).mean()
    return df_daily_averaged_sentiment_score


def get_topic_aggregated(df):
    news_emb = df_news.News.apply(lambda x: ft_model.get_sentence_vector(
        (x))).to_numpy().tolist()
    news_emb = np.array(news_emb)

    topic = lda_model.transform(ft_scaler.transform(news_emb))
    for i in range(5):
        df[f"Topic{i+1}"] = topic[:, i]
    df_daily_averaged_topic = df.groupby(['Date']).mean()
    return df_daily_averaged_topic


df_topic = get_topic_aggregated(df_news[["Date"]])
df_sentiment = get_sentiment_aggregated(df_news[["Date"]])
df_feature = pd.concat([df_sentiment, df_topic], axis=1)
df_feature.to_pickle(
    f"{HOME}/dev/features/RedditNews_2008-06-09_2016-07-01_4_features.pkl")