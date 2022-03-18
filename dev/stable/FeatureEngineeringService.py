from FeatureProviderFactory import FeatureProviderFactory


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
        self.idx = None

    def get_feature(self):
        df_news_feature = self.news_feature_helper.get_feature()
        df_price_feature, df_dt = self.price_feature_helper.get_feature()
        self.raw_X, self.y, self.idx = self.strategy.feature_extraction(
            df_news_feature, df_price_feature, df_dt, self.past, self.h)
        self.X = self.strategy.feature_selection(self.raw_X, self.y)
        self.feature_selector = self.strategy.feature_selector
        return self.X, self.y
