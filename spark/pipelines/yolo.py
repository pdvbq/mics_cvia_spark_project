from ultralytics import YOLO
from spark.pipelines.pipeline import Pipeline


class YoloPipeline(Pipeline):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = YOLO(self.model_path)

    def train(self, **kwargs):
        self.model.train(data=kwargs["data"], epochs=1)
        self.model.save("trained.pt")

    def test(self, **kwargs):
        results = self.model.val(data=kwargs["data"])
        print(results)

    def validate(self, **kwargs):
        print(self.model.val())
