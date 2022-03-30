from LinearModelFactory import LinearModelFactory
from ComponentModel import ComponentModel


class LinearComponentModel(ComponentModel):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.model = None

    def set_model(self, model_name="LR"):
        self.model = LinearModelFactory.get_model(model_name)

    def train(self):
        train_X = self.dataset.train_X
        train_y = self.dataset.train_y.ravel()
        self.model.fit(train_X, train_y)

    def predict(self, feature):
        return self.model.predict(feature)
