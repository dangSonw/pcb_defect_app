import numpy as np

from utils.vision_slicer import infer_with_optional_slicing


class DummyBoxes:
    def __init__(self):
        self.xyxy = [[1.0, 2.0, 3.0, 4.0]]
        self.cls = [0]
        self.conf = [0.9]


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()


class DummyModel:
    def __init__(self):
        self.calls = []

    def predict(self, source, conf=None, verbose=False, **kwargs):
        self.calls.append({"source": source, "conf": conf, "verbose": verbose, "kwargs": kwargs})
        return [DummyResult()]


def test_infer_with_optional_slicing_uses_conf_only():
    model = DummyModel()
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    detections = infer_with_optional_slicing(model, image, conf=0.3)

    assert len(detections) == 1
    assert detections[0][4] == 0
    assert len(model.calls) == 1
    assert model.calls[0]["conf"] == 0.3
    assert "iou" not in model.calls[0]["kwargs"]
