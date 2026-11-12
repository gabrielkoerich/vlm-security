from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class RandomForestClassifier:
    classes_: NDArray[np.str_]
    def __init__(
        self,
        n_estimators: int = 100,
        *,
        class_weight: Literal["balanced"] | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...
