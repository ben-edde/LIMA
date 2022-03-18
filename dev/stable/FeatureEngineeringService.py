class FeatureEngineeringService:
    def __init__(self, strategy) -> None:
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
        self.strategy.get_feature()
        self.raw_X, self.y, self.idx = self.strategy.feature_extraction(
            self.past, self.h)
        self.X = self.strategy.feature_selection(self.raw_X, self.y)
        self.feature_selector = self.strategy.feature_selector
        self.df_shifted = self.strategy.df_shifted
        return self.X, self.y
