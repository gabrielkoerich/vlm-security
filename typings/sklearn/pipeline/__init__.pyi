from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Pipeline:
    classes_: NDArray[np.str_]
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...

def make_pipeline(*steps: StandardScaler | LogisticRegression) -> Pipeline: ...
