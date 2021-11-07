import pandas as pd
import numpy as np
from openie import StanfordOpenIE
from gensim.models import Word2Vec
from sklearn.model_selection import cross_validate,RepeatedKFold
from sklearn.linear_model import Lasso
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag


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

df_news = pd.read_csv("RedditNews_2008-06-09_2016-07-01.csv")
df_price = pd.read_csv("WTI_Spot_2008-06-09_2016-07-01.csv")
df_price.Date = pd.to_datetime(df_price.Date)

sample=df_news.iloc[0].News
openie_client = StanfordOpenIE()


df_news.News=df_news.News.apply(lambda r: " ".join(clean(r)))

event_tuples = []
for idx, row in df_news.iterrows():
    text = row['News']
    for triple in openie_client.annotate(text):
        triple['Date'] = row['Date']
        event_tuples.append(triple)



df_events = pd.DataFrame(event_tuples)
df_events.to_csv("RedditNews_2008-06-09_2016-07-01_event_tuple.csv",index=False)
