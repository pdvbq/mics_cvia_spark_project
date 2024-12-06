from abc import abstractmethod


class Pipeline:
    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    @abstractmethod
    def validate(self, **kwargs):
        pass
