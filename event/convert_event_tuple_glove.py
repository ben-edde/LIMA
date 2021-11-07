import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from gensim.models import KeyedVectors

df_events = pd.read_csv("RedditNews_2008-06-09_2016-07-01_event_tuples.csv")
df_price = pd.read_csv("WTI_Spot_2008-06-09_2016-07-01.csv")

df_event_price = df_events.merge(df_price, on="Date", how="outer")
df_event_price = df_event_price[[
    "Date", "subject", "relation", "object", "Price"
]]

glove_file = 'gensim_glove.6B.100d.txt'
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


df_event_price.dropna(inplace=True)

df_event_price.subject = df_event_price.subject.apply(lambda x: get_glove(x))
df_event_price.relation = df_event_price.relation.apply(lambda x: get_glove(x))
df_event_price.object = df_event_price.object.apply(lambda x: get_glove(x))

df_event_price["event"] = df_event_price.apply(
    lambda x: np.array(x[1].tolist() + x[2].tolist() + x[3].tolist()), axis=1)
df_event_price.to_pickle(
    "df_RedditNews_2008-06-09_2016-07-01_event_tuples_glove.pkl")
