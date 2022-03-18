from sklearn.feature_selection import mutual_info_regression, RFE, RFECV, SelectFromModel, SequentialFeatureSelector, chi2, SelectKBest, f_regression, VarianceThreshold, r_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso
from FeatureProviderFactory import FeatureProviderFactory
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score


class FeatureEngineeringService:
    def __init__(self, strategy) -> None:
        self.news_feature_helper = FeatureProviderFactory.get_provider("news")
        self.price_feature_helper = FeatureProviderFactory.get_provider(
            "price")
        self.past = 25
        self.h = 1
        self.raw_X = None
        self.y = None
        self.X = None
        self.df_shifted = None
        self.feature_selector = None
        self.strategy = strategy

    def get_feature(self):
        df_news_feature = self.news_feature_helper.get_feature()
        df_price_feature, df_dt = self.price_feature_helper.get_feature()
        self.raw_X, self.y, self.idx = self.strategy.feature_extraction(
            df_news_feature, df_price_feature, df_dt, self.past, self.h)
        self.X = self.strategy.feature_selection(self.raw_X, self.y)
        self.feature_selector = self.strategy.feature_selector
        return self.X, self.y
