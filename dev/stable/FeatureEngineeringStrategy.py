from abc import ABC, abstractmethod


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def feature_extraction(self):
        pass
    @abstractmethod
    def feature_selection(self):
        pass
class ModelBuildingFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def feature_extraction(df_news_feature,df_price_feature):
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

class ForecastFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def feature_extraction(self):
        df_news_feature=self.news_feature_helper.get_feature()
        df_price_feature,df_dt=self.price_feature_helper.get_feature()
        df_Xy = pd.concat([df_news_feature, df_price_feature], axis=1, join="inner")
        # TODO add filter and modify following
        # df_shifted = self.series_to_supervised(df_Xy.dropna(), self.past, self.h)

        # # remove current day features for forecast
        # for each in df_shifted.columns[:-1]:
        #     if "(t)" in each:
        #         df_shifted.drop(each, axis=1, inplace=True)
        # # add time feature without shift 
        # self.df_shifted=pd.concat([df_dt,df_shifted],axis=1).dropna()
        # self.raw_X = self.df_shifted.to_numpy()[:, :-1]
        # self.y =  self.df_shifted.to_numpy()[:, -1].reshape(-1, 1)
    
    def feature_selection(self):
        # TODO load trained model selector for selection
        # estimator = Lasso(random_state=42)
        # self.feature_selector = RFE(estimator,n_features_to_select=10,step=1)
        # scaled_raw_X=MinMaxScaler().fit_transform(self.raw_X)
        # self.feature_selector = self.feature_selector.fit(scaled_raw_X, self.y.ravel())
        # self.X = self.raw_X[:, self.feature_selector.get_support()]
        pass