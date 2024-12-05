from abc import abstractmethod


class Pipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    @abstractmethod
    def validate(self, **kwargs):
        pass
