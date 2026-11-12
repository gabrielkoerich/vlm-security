from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from torch import cuda as cuda
from torch import mps as mps

class Tensor:
    def numpy(self) -> NDArray[np.float32]: ...
    def tolist(self) -> list[float]: ...

class no_grad:
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

def set_num_threads(n: int) -> None: ...
