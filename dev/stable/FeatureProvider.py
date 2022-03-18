from abc import abstractmethod


class FeatureProvider:
    @abstractmethod
    def get_raw_data(self, mode):
        raise NotImplementedError

    @abstractmethod
    def get_feature(self, mode):
        raise NotImplementedError
