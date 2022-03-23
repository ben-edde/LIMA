from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self,
                 X=None,
                 y=None,
                 idx=None,
                 train_ratio=1,
                 scaling=False,
                 feature_scaler=None,
                 label_scaler=None):
        self.feature = X
        self.label = y
        self.train_ratio = train_ratio
        self.scaling = scaling
        self.idx = idx
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.train_X = None
        self.train_y = None
        self.train_idx = None
        self.test_X = None
        self.test_y = None
        self.test_idx = None
        self.update()

    def set_train_ratio(self, train_ratio):
        self.train_ratio = train_ratio
        self.update()

    def update(self):
        self.length = self.feature.shape[0]
        train_size = int(self.length * self.train_ratio)
        self.train_X = self.feature[:train_size]
        self.test_X = self.feature[train_size:]
        self.train_idx = self.idx[:train_size]
        self.test_idx = self.idx[train_size:]
        if not self.label is None:
            self.train_y = self.label[:train_size]
            self.test_y = self.label[train_size:]
        if self.scaling:
            self.normalize()

    def set_feature_scaler(self, feature_scaler):
        self.feature_scaler = feature_scaler

    def get_feature_scaler(self):
        if self.feature_scaler is None:
            self.feature_scaler = MinMaxScaler()
        return self.feature_scaler

    def set_label_scaler(self, label_scaler):
        self.label_scaler = label_scaler

    def get_label_scaler(self):
        if self.label_scaler is None:
            self.label_scaler = MinMaxScaler(feature_range=(1, 100))
        return self.label_scaler

    def normalize(self):
        self.feature_scaler = self.get_feature_scaler()
        self.feature_scaler.fit(self.train_X)
        self.train_X = self.feature_scaler.transform(self.train_X)
        if len(self.test_X) > 0:
            self.test_X = self.feature_scaler.transform(self.test_X)

        if not self.label is None:
            self.label_scaler = self.get_label_scaler()
            self.label_scaler.fit(self.train_y)
            self.train_y = self.label_scaler.transform(self.train_y)
            if len(self.test_y) > 0:
                self.test_y = self.label_scaler.transform(self.test_y)

    def get_TS_cv(self, k=5, test_size=None):
        return TimeSeriesSplit(
            n_splits=k,
            gap=0,
            test_size=test_size,
        )
