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
from sklearn.feature_extraction.text import CountVectorizer

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

HOME = os.environ['LIMA_HOME']
df_news = pd.read_csv(f"{HOME}/data/reuse/RedditNews_WTI/RedditNews_2008-06-09_2016-07-01.csv")
df_news.News = df_news.News.apply(lambda r: " ".join(clean(r)))

lda=LatentDirichletAllocation(n_components=4,random_state=42)


vectorizer = CountVectorizer()
tf = vectorizer.fit_transform(df_news.News)
lda.fit(tf)
topic_series=df_news.News.apply(lambda x: np.mean(lda.transform(vectorizer.transform(x.split(" "))),axis=0).tolist())
df_topic=pd.DataFrame(topic_series.to_list())
df=pd.concat([df_news,df_topic],axis=1)
df=df.drop("News",axis=1)
df.columns=["Date","Topic1","Topic2","Topic3","Topic4"]
df.to_pickle(
    "RedditNews_2008-06-09_2016-07-01_4_topic.pkl")
df.groupby("Date").mean().to_pickle(
    "RedditNews_2008-06-09_2016-07-01_4_topic_daily.pkl")
