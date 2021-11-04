import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from framework import run_experiment
from sacred import Experiment
from sacred.observers import FileStorageObserver
ex = Experiment("mocking experiment")
ex.observers.append(FileStorageObserver('runs'))


def mocked_feature_extraction(df_news):
    df_result=df_news[["date"]]
    df_result["polarity"]=df_news.apply(lambda row:TextBlob(row['news']).sentiment.polarity,axis=1)
    df_daily_averaged_sentiment_score=df_result.groupby(['date']).mean()
    # df_daily_averaged_sentiment_score["feature_2"]=df_daily_averaged_sentiment_score["polarity"]*0.5
    # df_daily_averaged_sentiment_score["feature_3"]=df_daily_averaged_sentiment_score["polarity"]*0.75
    # df_daily_averaged_sentiment_score["feature_4"]=df_daily_averaged_sentiment_score["polarity"]*0.25
    return df_daily_averaged_sentiment_score

def mocked_model():
    return AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
        splitter="random", max_depth=1, min_samples_split=3),
                            n_estimators=30,
                            learning_rate=0.01)
@ex.automain
def main():
    run_experiment(feature_method=mocked_feature_extraction,model_method=mocked_model,description="demo experiment",ratio=0.05,exp=ex)

