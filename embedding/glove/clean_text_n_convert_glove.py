import os
import string

import fasttext
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

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


df_price = pd.read_csv(f"{HOME}/embedding/WTI_Spot_2008-06-09_2016-07-01.csv")
df_news = pd.read_csv(f"{HOME}/embedding/RedditNews_2008-06-09_2016-07-01.csv")

from gensim.models import KeyedVectors
from nltk.stem.porter import PorterStemmer

glove_file = f'{HOME}/embedding/glove/gensim_glove.6B.300d.txt'
model = KeyedVectors.load_word2vec_format(glove_file, binary=False)
porter = PorterStemmer()


def get_glove(word):
    word = word.split(" ")[0] if len(word.split(" ")) > 1 else word
    try:
        return model.get_vector(word)
    except:
        try:
            return model.get_vector(porter.stem(word))
        except:
            # return np.zeros((100))    # may be dangerous to use 0, drop for safety
            return None


df_news.News = df_news.News.apply(lambda r: " ".join(clean(r)))
df_news_price = df_news.merge(df_price, on="Date", how="outer")
df_news_price.dropna(inplace=True)

df_news_price["News_glove"] = df_news_price.News.apply(lambda x: get_glove(x))
df_news_price.dropna(inplace=True)

df_news_price = df_news_price[["Date", "News_glove", "Price"]]
df_news_price.to_pickle(
    "WTI_Spot_n_RedditNews_2008-06-09_2016-07-01_glove_300d.pkl")
