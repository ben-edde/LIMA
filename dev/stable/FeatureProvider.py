from abc import abstractmethod


class FeatureProvider:
    @abstractmethod
    def get_raw_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_feature(self):
        raise NotImplementedError
