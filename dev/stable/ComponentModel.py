from abc import abstractclassmethod


class ComponentModel:
    def __init__(self) -> None:
        self.dataset = None

    @abstractclassmethod
    def set_model(self, model_name: str):
        pass

    @abstractclassmethod
    def train(self):
        pass

    @abstractclassmethod
    def predict(self):
        pass