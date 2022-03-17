import datetime
import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
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
from scipy import spatial
from FeatureProvider import FeatureProvider

HOME = os.environ['LIMA_HOME']


class NewsFeatureProvider(FeatureProvider):
    def __init__(self) -> None:
        self.fasttext_model = fasttext.load_model(
            f"{HOME}/data/big/cc.en.300.bin")
        self.emb_scaler = None
        self.lda_model = None

    def get_raw_data(self):
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        mongo_db = mongo_client["lima"]
        mongo_collection = mongo_db["investing_news"]
        cursor = mongo_collection.find({
            "News": {
                "$ne": "NEURONswap: First Dex To Implement Governance 2.0"
            }
        })
        df_news = pd.DataFrame(list(cursor))[["Date", "News"]]
        return df_news

    def clean_text(self, text: str) -> list:
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

    def clean_news(self, df):
        df.News = df.News.apply(lambda r: " ".join(self.clean_text(r)))
        return df

    def get_sentiment_aggregated(self, df):
        df = df.copy()
        df["Polarity"] = df.apply(
            lambda row: TextBlob(row['News']).sentiment.polarity, axis=1)
        df["Subjectivity"] = df.apply(
            lambda row: TextBlob(row['News']).sentiment.subjectivity, axis=1)
        df_daily_averaged_sentiment_score = df.groupby(['Date']).mean()
        return df_daily_averaged_sentiment_score

    def get_topic_aggregated(self, df):
        df = df.copy()
        news_emb = df.News.apply(lambda x: self.fasttext_model.
                                 get_sentence_vector((x))).to_numpy().tolist()
        news_emb = np.array(news_emb)
        self.emb_scaler = MinMaxScaler()
        news_emb = self.emb_scaler.fit_transform(news_emb)
        self.lda_model = LatentDirichletAllocation(n_components=5, n_jobs=-1)
        topic = self.lda_model.fit_transform(news_emb)
        for i in range(5):
            df[f"Topic{i+1}"] = topic[:, i]
        df_daily_averaged_topic = df.groupby(['Date']).mean()
        return df_daily_averaged_topic

    def get_geoidx_aggregated(self, df):
        geo_pattern = {
            "Geopolitical_Threats":
            "Geopolitical risk concern tension uncertainty United States tensions military war geopolitical coup guerrilla warfare Latin America Central America South America Europe Africa Middle East Far East Asia",
            "Nuclear_Threats":
            "nuclear war atomic war nuclear conflict atomic conflict nuclear missile fear threat risk peril menace",
            "War_Threats":
            "war risk risk of war fear of war war fear military threat war threat threat of war military action military operation military fce risk threat",
            "Terrorist_Threats":
            "terrorist threat threat of terrorism terrorism menace menace of terrorism terrorist risk terr risk risk of terrorism terr threat",
            "War_Acts":
            "beginning of the war outbreak of the war onset of the war escalation of the war start of the war war military air strike war battle heavy casualties",
            "Terrorist_Acts": "terrorist act terrorist acts"
        }
        df = df.copy()
        df["news_emb"] = df.News.apply(
            lambda x: self.fasttext_model.get_sentence_vector(
                (x))).to_numpy().tolist()
        for each in geo_pattern:
            pattern_emb = self.fasttext_model.get_sentence_vector(each)
            df[each] = df.news_emb.apply(
                lambda x: 1 - spatial.distance.cosine(pattern_emb, x))
        df = df.drop(["News", "news_emb"], axis=1)
        df_daily_averaged_geoidx = df.groupby(['Date']).max()
        return df_daily_averaged_geoidx

    def decay_features(self, df):
        window_size = 5
        feature_list = []
        for each in df.columns:
            feature_list.append(df[each].iloc[:window_size].to_list())
        feature_num = len(feature_list)
        for idx in range(window_size, len(df)):
            feature_tmp = np.zeros(feature_num)
            for t in range(window_size):
                for feature_idx in range(feature_num):
                    feature_tmp[feature_idx] += df.iloc[idx - t][
                        df.columns[feature_idx]] * (
                            (window_size - t) / window_size)
            for feature_idx in range(feature_num):
                feature_list[feature_idx].append(feature_tmp[feature_idx])
        df_result = pd.DataFrame(feature_list).transpose()
        df_result.index = df.index
        df_result.columns = [f"Decay_{each}" for each in df.columns]
        return df_result

    def get_feature(self):
        df_news = self.get_raw_data()
        df_news = self.clean_news(df_news)

        # sentiment features
        df_sentiment = self.get_sentiment_aggregated(df_news)
        df_sentiment["Combined_Sentiment"] = df_sentiment.Polarity * (
            1 + df_sentiment.Subjectivity)
        df_decay = self.decay_features(df_sentiment)
        df_sentiment = pd.concat([df_sentiment, df_decay], axis=1)
        # topic features
        df_topic = self.get_topic_aggregated(df_news)
        df_decay = self.decay_features(df_topic)
        df_topic = pd.concat([df_topic, df_decay], axis=1)
        # geo risk features
        df_geo_risk = self.get_geoidx_aggregated(df_news)
        df_decay = self.decay_features(df_geo_risk)
        df_geo_risk = pd.concat([df_geo_risk, df_decay], axis=1)

        df_feature = pd.concat([df_sentiment, df_topic, df_geo_risk], axis=1)
        return df_feature