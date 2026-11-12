from torch import Tensor

class Matrix:
    def tolist(self) -> list[list[float]]: ...

class Boxes:
    cls: Tensor
    conf: Tensor
    xywh: Matrix

class Results:
    path: str
    boxes: Boxes | None

class YOLO:
    def __init__(self, model: str) -> None: ...
    def predict(
        self,
        source: list[str],
        *,
        imgsz: int,
        conf: float,
        device: str,
        verbose: bool,
    ) -> list[Results]: ...
