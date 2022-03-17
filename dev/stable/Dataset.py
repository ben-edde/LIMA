from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, X=None, y=None, train_ratio=1, scaling=False):
        self.feature = X
        self.label = y
        self.train_ratio = train_ratio
        self.scaling = scaling
        self.update()

    def set_train_ratio(self, train_ratio):
        self.train_ratio = train_ratio
        self.update()

    def update(self):
        self.length = self.feature.shape[0]
        train_size = int(self.length * self.train_ratio)
        self.train_X = self.feature[:train_size]
        self.train_y = self.label[:train_size]
        self.test_X = self.feature[train_size:]
        self.test_y = self.label[train_size:]
        if self.scaling:
            self.normalize()

    def normalize(self):
        self.feature_scaler = MinMaxScaler()
        self.feature_scaler.fit(self.train_X)
        self.train_X = self.feature_scaler.transform(self.train_X)
        self.test_X = self.feature_scaler.transform(self.test_X)

        self.label_scaler = MinMaxScaler(feature_range=(1, 100))
        self.label_scaler.fit(self.train_y)
        self.train_y = self.label_scaler.transform(self.train_y)
        self.test_y = self.label_scaler.transform(self.test_y)

    def get_TS_cv(self, k=5, test_size=None):
        return TimeSeriesSplit(
            n_splits=k,
            gap=0,
            test_size=test_size,
        )