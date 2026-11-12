from typing import Self

from PIL.Image import Image
from torch import Tensor

class BatchFeature(dict[str, Tensor]): ...

class BaseModelOutputWithPooling:
    pooler_output: Tensor

class AutoImageProcessor:
    @classmethod
    def from_pretrained(cls, name: str) -> Self: ...
    def __call__(self, *, images: list[Image], return_tensors: str) -> BatchFeature: ...

class PreTrainedModel:
    def eval(self) -> Self: ...
    def __call__(self, **inputs: Tensor) -> BaseModelOutputWithPooling: ...

class AutoModel:
    @classmethod
    def from_pretrained(cls, name: str) -> PreTrainedModel: ...
