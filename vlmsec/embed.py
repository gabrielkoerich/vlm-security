"""Image embeddings for the hybrid score, from a frozen DINOv2-small.

torch and transformers are imported when the first crop is embedded, not at import time, so the
bridge, the tests and the box model never pay for them. They come from the `vision` dependency
group: `uv sync --group vision`.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image

if TYPE_CHECKING:
    from transformers import AutoImageProcessor, PreTrainedModel

MODEL = "facebook/dinov2-small"
BATCH = 32
# the detector and the VLM share this machine, two threads embed a crop well under a second
THREADS = 2


class Dinov2:
    """Callable as a hybrid.Embedder. Loads the weights on the first call and keeps them."""

    def __init__(self, name: str = MODEL) -> None:
        self.name: str = name
        self._proc: AutoImageProcessor | None = None
        self._net: PreTrainedModel | None = None

    def _load(self) -> tuple["AutoImageProcessor", "PreTrainedModel"]:
        if self._proc is None or self._net is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            torch.set_num_threads(THREADS)
            self._proc = AutoImageProcessor.from_pretrained(self.name)
            self._net = AutoModel.from_pretrained(self.name).eval()
        return self._proc, self._net

    def __call__(self, images: Sequence[Image]) -> NDArray[np.float32]:
        import torch

        proc, net = self._load()
        out: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for i in range(0, len(images), BATCH):
                batch = proc(images=list(images[i:i + BATCH]), return_tensors="pt")
                out.append(net(**batch).pooler_output.numpy())
        return np.concatenate(out)
