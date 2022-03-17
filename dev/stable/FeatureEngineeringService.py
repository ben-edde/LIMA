from sklearn.feature_selection import mutual_info_regression,RFE,RFECV,SelectFromModel,SequentialFeatureSelector,chi2,SelectKBest,f_regression,VarianceThreshold,r_regression
from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.svm import LinearSVR,SVR
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
    def __init__(self) -> None:
        self.news_feature_helper=FeatureProviderFactory.get_provider("news")
        self.price_feature_helper=FeatureProviderFactory.get_provider("price")
        self.past=25
        self.h=1
        self.raw_X=None
        self.y=None
        self.X=None
        self.df_shifted=None
        self.feature_selector=None
        
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

    def feature_extraction(self):
        df_news_feature=self.news_feature_helper.get_feature()
        df_price_feature,df_dt=self.price_feature_helper.get_feature()
        df_Xy = pd.concat([df_news_feature, df_price_feature], axis=1, join="inner")
        
        df_shifted = self.series_to_supervised(df_Xy.dropna(), self.past, self.h)

        # remove current day features for forecast
        for each in df_shifted.columns[:-1]:
            if "(t)" in each:
                df_shifted.drop(each, axis=1, inplace=True)
        # add time feature without shift 
        self.df_shifted=pd.concat([df_dt,df_shifted],axis=1).dropna()
        self.raw_X = self.df_shifted.to_numpy()[:, :-1]
        self.y =  self.df_shifted.to_numpy()[:, -1].reshape(-1, 1)
    
    def feature_selection(self):
        estimator = Lasso(random_state=42)
        self.feature_selector = RFE(estimator,n_features_to_select=10,step=1)
        scaled_raw_X=MinMaxScaler().fit_transform(self.raw_X)
        self.feature_selector = self.feature_selector.fit(scaled_raw_X, self.y.ravel())
        self.X = self.raw_X[:, self.feature_selector.get_support()]

    def get_feature_label(self):
        self.feature_extraction()
        self.feature_selection()
        return self.X, self.y
        

