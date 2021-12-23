import pandas as pd
import numpy as np
import fasttext
import os

HOME = os.environ['LIMA_HOME']

df_events = pd.read_csv(
    f"{HOME}/data/reuse/RedditNews_WTI/RedditNews_2008-06-09_2016-07-01_event_tuples.csv"
)

model = fasttext.load_model(f'{HOME}/data/big/cc.en.300.bin')

df_events.subject = df_events.subject.apply(
    lambda x: model.get_sentence_vector((x)))
df_events.relation = df_events.relation.apply(
    lambda x: model.get_sentence_vector((x)))
df_events.object = df_events.object.apply(lambda x: model.get_sentence_vector(
    (x)))
df_events.dropna(inplace=True)

df_events.to_pickle(
    "RedditNews_2008-06-09_2016-07-01_event_tuples_fasttext.pkl")
df_events.groupby("Date").mean().to_pickle(
    "RedditNews_2008-06-09_2016-07-01_event_tuples_fasttext_daily.pkl")
