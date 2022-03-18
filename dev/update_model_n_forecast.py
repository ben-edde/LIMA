import datetime
import logging
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.layers import Reshape, MaxPool3D, Bidirectional, ConvLSTM2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Conv2D
from keras.layers import Dropout
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso
import pymongo
import random
import string
import fasttext
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

        names += [f'{data.columns[j]}(t-{i})' for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{data.columns[j]}(t)' for j in range(n_vars)]
        else:
            names += [f'{data.columns[j]}(t+{i})' for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



# set random seed
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
h = 1
past = 10

from influxdb_client import InfluxDBClient
client= InfluxDBClient.from_config_file(f"{HOME}/dev/DB/influxdb_config.ini")
query_api = client.query_api()
df_WTI = query_api.query_data_frame("""
from(bucket: "dummy")
  |> range(start: -20d, stop: now())
  |> filter(fn: (r) => r["_measurement"] == "WTI") 
  |> filter(fn: (r) => r["type"] == "closing_price") 
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> drop(columns: ["_start", "_stop"])
""")

df_WTI=df_WTI[["_time","CLC4","CLC3","CLC2","CLC1"]]
df_WTI.columns=["Date","CLC4","CLC3","CLC2","CLC1"]
df_WTI.set_index("Date",inplace=True)
df_WTI.index=df_WTI.index.map(lambda each: each.date())
df_WTI.index=pd.to_datetime(df_WTI.index)


mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["lima"]
mongo_collection = mongo_db["investing_news"]
cursor = mongo_collection.find({"News":{"$ne":"NEURONswap: First Dex To Implement Governance 2.0"}})
df_news =  pd.DataFrame(list(cursor))[["Date","News"]]
df_news=df_news[df_news.Date.isin(df_WTI.index)].reset_index(drop=True)

df_news.News = df_news.News.apply(lambda r: " ".join(clean(r)))
fasttext_model = fasttext.load_model(f"{HOME}/data/big/cc.en.300.bin")
trained_model = keras.models.load_model(f"{HOME}/dev/models/investing/GRU.model")
trained_X_scaler=joblib.load(f"{HOME}/dev/models/investing/feature_scaler(110).joblib")
trained_y_scaler=joblib.load(f"{HOME}/dev/models/investing/label_scaler(1).joblib")
trained_feature_selector=joblib.load(f"{HOME}/dev/models/investing/feature_selector(30).joblib")
trained_lda_model=joblib.load(f"{HOME}/dev/models/investing/lda_model.joblib")
trained_emb_scaler=joblib.load(f"{HOME}/dev/models/investing/emb_scaler.joblib")

def get_sentiment_features(df):
    df=df.copy()
    df["Polarity"] = df.apply(
        lambda row: TextBlob(row['News']).sentiment.polarity, axis=1)
    df["Subjectivity"] = df.apply(
        lambda row: TextBlob(row['News']).sentiment.subjectivity, axis=1)
    df_daily_averaged_sentiment_score = df.groupby(['Date']).mean()
    return df_daily_averaged_sentiment_score

def get_topic_features(df, emb_scaler, lda_model):
    df=df.copy()
    news_emb = df.News.apply(lambda x: fasttext_model.get_sentence_vector(
        (x))).to_numpy().tolist()
    news_emb = np.array(news_emb)
    news_emb=emb_scaler.fit_transform(news_emb)
    topic= lda_model.fit_transform(news_emb)
    for i in range(5):
        df[f"Topic{i+1}"] = topic[:, i]
    df_daily_averaged_topic = df.groupby(['Date']).mean()
    return df_daily_averaged_topic
df_topic= get_topic_features(df_news,trained_emb_scaler,trained_lda_model)
df_sentiment = get_sentiment_features(df_news)
df_features = pd.concat([df_sentiment, df_topic], axis=1)

df_Xy=pd.concat([df_features,df_WTI],axis=1,join="inner")

# 1st order DIFF
df_original_price=df_Xy[["CLC1"]].shift(h).dropna()
df_Xy=df_Xy.diff().dropna()

# shift back $past days
df_Xy=series_to_supervised(df_Xy,past,h)

df_original_price=df_original_price[df_original_price.index.isin(df_Xy.index)]
# remove current day features for forecast
for each in df_Xy.columns[:-1]:
    if "(t)" in each:
        df_Xy.drop(each,axis=1,inplace=True)
raw_X=df_Xy.to_numpy()[:,:-1]
y=df_Xy.to_numpy()[:,-1].reshape(-1, 1)

X=trained_X_scaler.fit_transform(raw_X)
X=X[:,trained_feature_selector.get_support()]
y=trained_y_scaler.fit_transform(y)

from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.00001)
trained_model.compile(loss='mse', optimizer=opt)
history = trained_model.fit(X,
                    y,
                    epochs=50,
                    batch_size=10,
                    verbose=1,
                    shuffle=False)

df_inf_Xy=pd.concat([df_features,df_WTI],axis=1,join="inner")
# no shift: pred of t+1 use t as t-1
df_inf_original_price=df_inf_Xy[["CLC1"]].dropna()

df_inf_Xy=df_inf_Xy.diff().dropna()
# shift past-1: past 9 + current day
df_inf_Xy=series_to_supervised(df_inf_Xy,past-1,1)

df_inf_original_price = df_inf_original_price[df_inf_original_price.index.isin(
    df_inf_Xy.index)]

df_inf_Xy.index=df_inf_Xy.index.map(lambda x: x+datetime.timedelta(days=1))
df_inf_original_price.index=df_inf_original_price.index.map(lambda x: x+datetime.timedelta(days=1))

inf_raw_X=df_inf_Xy.to_numpy()
inf_raw_X=trained_X_scaler.transform(inf_raw_X)
inf_X=inf_raw_X[:,trained_feature_selector.get_support()]

inf_y=trained_model.predict(inf_X)
inverted_inf_y=trained_y_scaler.transform(inf_y) +df_inf_original_price.to_numpy()
df_inf=pd.DataFrame(inverted_inf_y,columns=["CLC1_forecast"])
df_inf.index=df_inf_original_price.index
df_inf.index=df_inf.index.map(lambda x: datetime.datetime.combine(x,datetime.time(22,0)))
def shift_weekend(x):
    if x.weekday()in [5,6]:
        x=x+datetime.timedelta(days=7-x.weekday())
    return x
df_inf.index=df_inf.index.map(shift_weekend)
df_inf.columns=["CLC1"]
df_inf["h"]=1
df_inf["type"]="forecast_o"


from influxdb_client.client.write_api import SYNCHRONOUS,ASYNCHRONOUS
client= InfluxDBClient.from_config_file(f"{HOME}/dev/DB/influxdb_config.ini")
write_api=client.write_api(write_options=SYNCHRONOUS)
write_api.write(bucket="dummy",record=df_inf, data_frame_measurement_name='WTI',data_frame_tag_columns=["h","type"])
write_api.close()
client.close()


trained_model.save(f"{HOME}/dev/models/investing/GRU.model")
joblib.dump(trained_X_scaler,f"{HOME}/dev/models/investing/feature_scaler(110).joblib")
joblib.dump(trained_y_scaler,f"{HOME}/dev/models/investing/label_scaler(1).joblib")
joblib.dump(trained_lda_model,f"{HOME}/dev/models/investing/lda_model.joblib")
joblib.dump(trained_emb_scaler,f"{HOME}/dev/models/investing/emb_scaler.joblib")