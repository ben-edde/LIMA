# Import libraries
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import string
from openie import StanfordOpenIE
from gensim.models import Word2Vec

df_news = pd.read_csv("RedditNews.csv")

df_news = df_news[:5000]

openie_client = StanfordOpenIE()

event_tuples = []
for idx, row in df_news.iterrows():
    text = row['News']
    for triple in openie_client.annotate(text):
        triple['Date'] = row['Date']
        event_tuples.append(triple)


df_events = pd.DataFrame(event_tuples)

#df_events.to_csv("event_tuples_sample.csv", index=False)

event_subject = df_events.subject.apply(lambda x: x.split(" "))
event_relation = df_events.relation.apply(lambda x: x.split(" "))
event_object = df_events.object.apply(lambda x: x.split(" "))

w2v_model = Word2Vec(event_subject, min_count=1)

w2v_model.build_vocab(event_subject, update=True)
w2v_model.build_vocab(event_relation, update=True)
w2v_model.build_vocab(event_object, update=True)

w2v_model.train(event_subject,
                total_examples=len(event_subject),
                epochs=30,
                report_delay=1)
w2v_model.train(event_relation,
                total_examples=len(event_relation),
                epochs=30,
                report_delay=1)
w2v_model.train(event_object,
                total_examples=len(event_object),
                epochs=30,
                report_delay=1)


# if more than 1 word, only pick 1st one
df_events.subject = df_events.subject.apply(lambda x: x.split(" ")[0]
                                            if len(x.split(" ")) > 1 else x)
df_events.relation = df_events.relation.apply(lambda x: x.split(" ")[0]
                                              if len(x.split(" ")) > 1 else x)
df_events.object = df_events.object.apply(lambda x: x.split(" ")[0]
                                          if len(x.split(" ")) > 1 else x)

df_events.subject = df_events.subject.apply(lambda x: w2v_model.wv[x])
df_events.relation = df_events.relation.apply(lambda x: w2v_model.wv[x])
df_events.object = df_events.object.apply(lambda x: w2v_model.wv[x])

df_events.to_pickle("event_tuple_w2v.pkl")
 