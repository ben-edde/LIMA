from abc import ABC, abstractmethod

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineeringStrategy:
    @abstractmethod
    def feature_extraction(self, df_news_feature, df_price_feature, past, h):
        pass

    @abstractmethod
    def feature_selection(self, raw_X, y):
        pass

    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
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


class ModelBuildingFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def feature_extraction(self, df_news_feature, df_price_feature, df_dt,
                           past, h):
        df_Xy = pd.concat([df_news_feature, df_price_feature],
                          axis=1,
                          join="inner")
        df_shifted = self.series_to_supervised(df_Xy.dropna(), past, h)

        # remove current day features for forecast
        for each in df_shifted.columns[:-1]:
            if "(t)" in each:
                df_shifted.drop(each, axis=1, inplace=True)
        # add time feature without shift
        self.df_shifted = pd.concat([df_dt, df_shifted], axis=1).dropna()
        self.raw_X = self.df_shifted.to_numpy()[:, :-1]
        self.y = self.df_shifted.to_numpy()[:, -1].reshape(-1, 1)
        return self.raw_X, self.y, self.df_shifted.index

    def feature_selection(self, raw_X, y):
        estimator = Lasso(random_state=42)
        self.feature_selector = RFE(estimator, n_features_to_select=10, step=1)
        scaled_raw_X = MinMaxScaler().fit_transform(raw_X)
        self.feature_selector = self.feature_selector.fit(
            scaled_raw_X, y.ravel())
        self.X = raw_X[:, self.feature_selector.get_support()]
        return self.X


class ForecastFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def __init__(self):
        self.feature_selector = None

    def feature_extraction(self, df_news_feature, df_price_feature, df_dt,
                           past, h):
        df_Xy = pd.concat([df_news_feature, df_price_feature],
                          axis=1,
                          join="inner")
        df_shifted = self.series_to_supervised(df_Xy, past - 1, 1)
        self.df_shifted = pd.concat([df_dt, df_shifted], axis=1).dropna()
        return self.df_shifted.to_numpy(), None, self.df_shifted.index

    def get_feature_selector(self):
        return self.feature_selector

    def set_feature_selector(self, feature_selector):
        self.feature_selector = feature_selector

    def feature_selection(self, raw_X, y):
        feature_selector = self.get_feature_selector()
        self.X = feature_selector.transform(raw_X)
        return self.X
