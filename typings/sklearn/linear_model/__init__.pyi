from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class LogisticRegression:
    classes_: NDArray[np.str_]
    def __init__(
        self,
        *,
        C: float = 1.0,
        class_weight: Literal["balanced"] | None = None,
        max_iter: int = 100,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...
