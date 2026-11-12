"""The hybrid score: how likely a detection is a real subject, from its box and from its pixels.

Two small models, averaged:

  * p_box, random forests on box features: confidence, size, area, position, travel, camera,
    class and hour. Cheap, and good at "this camera always has a box of that size there".
  * p_emb, a logistic regression on an image embedding of a square crop around the box,
    1.5 times the box's longer side so some ground stays in the crop. The embedding comes from a
    frozen DINOv2-small (embed.py). Good at "that is a plant, whatever the box says".

p = (p_box + p_emb) / 2, then three bands (see `route`):

  * p < LOW (0.1): agrees with the scenery gate, the model is not asked
  * p >= HIGH (0.9): sure it is real, the model is skipped and geometry places it
  * in between: the model is asked, as before

Both models are trained from the bridge's own logged decisions, see train.py. Without a bundle on
disk every score is None and the pipeline behaves like the original MVP: the gate decides
scenery and the model is asked about everything else.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import sklearn
from numpy.typing import NDArray
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from vlmsec import config
from vlmsec.window import Box

LOW = config.get_float("HYBRID_LOW", 0.1)
HIGH = config.get_float("HYBRID_HIGH", 0.9)
CROP_MARGIN = 1.5
CROP_MIN = 64
BUNDLE = Path(config.get("HYBRID_BUNDLE", str(config.RUN_DIR / "hybrid.joblib")))
CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog")
# Classes a sure call may name without the model. The detector cannot tell everything else apart
SURE_CLASSES = frozenset(CLASSES)

type Embedder = Callable[[Sequence[Image.Image]], NDArray[np.float32]]
type Route = Literal["scenery", "ask", "sure"]


@dataclass(frozen=True)
class Event:
    """What the box model sees of one detection."""

    camera: int
    cls: str
    conf: float
    box: Box
    moved: float
    hour: int


def features(e: Event, cameras: Sequence[int]) -> list[float]:
    cx, cy, w, h = e.box
    one_hot_cam = [1.0 if e.camera == c else 0.0 for c in cameras]
    one_hot_cls = [1.0 if e.cls == c else 0.0 for c in CLASSES]
    return [e.conf, w, h, w * h, cx, cy, e.moved, float(e.hour), *one_hot_cam, *one_hot_cls]


def crop(img: Image.Image, box: Box, margin: float = CROP_MARGIN) -> Image.Image:
    """A square around the box, `margin` times its longer side, clamped to the frame."""
    cx, cy, w, h = box
    side = max(CROP_MIN, margin * max(w, h))
    x0, y0 = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
    return img.crop((int(x0), int(y0), int(min(img.width, x0 + side)), int(min(img.height, y0 + side))))


@dataclass
class Bundle:
    """Everything a score needs, written by train.py and read by `load`."""

    box: list[RandomForestClassifier]
    emb: Pipeline
    cameras: list[int]
    embedder: str
    sklearn: str
    labels: int


class BundleMismatch(RuntimeError):
    pass


def load(path: Path = BUNDLE) -> Bundle | None:
    """The bundle, or None when there is none. Refuses one pickled by another scikit-learn.

    A pickled forest does not load reliably across scikit-learn versions, and a model that loads
    wrong scores wrong without saying so. joblib runs code from the file on load, so only point
    HYBRID_BUNDLE at a bundle you trained yourself.
    """
    if not path.exists():
        return None
    got = joblib.load(path)
    if not isinstance(got, Bundle):
        raise BundleMismatch(f"{path} is not a hybrid bundle")
    if got.sklearn != sklearn.__version__:
        raise BundleMismatch(f"{path} was built with scikit-learn {got.sklearn}, this environment "
                             + f"has {sklearn.__version__}")
    return got


def save(bundle: Bundle, path: Path = BUNDLE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def _real(classes: NDArray[np.str_], proba: NDArray[np.float64]) -> float:
    """P(real) of a one-row prediction."""
    return float(proba[:, list(classes).index("real")].mean())


class Hybrid:
    def __init__(self, bundle: Bundle, embed: Embedder) -> None:
        self.bundle: Bundle = bundle
        self.embed: Embedder = embed

    def p_box(self, e: Event) -> float:
        x = [features(e, self.bundle.cameras)]
        return float(np.mean([_real(m.classes_, m.predict_proba(x)) for m in self.bundle.box]))

    def p_emb(self, img: Image.Image, box: Box) -> float:
        vec = self.embed([crop(img, box)])
        return _real(self.bundle.emb.classes_, self.bundle.emb.predict_proba(vec))

    def score(self, frame: Path, e: Event) -> float:
        with Image.open(frame) as raw:
            img = raw.convert("RGB")
        return round((self.p_box(e) + self.p_emb(img, e.box)) / 2, 3)


def route(gate_scenery: bool, p: float | None, cls: str) -> Route:
    """Where an event goes after the gate, given the hybrid's score (None when there is no model).

    The low band only confirms the gate. It never closes an event the gate let through, so a
    model trained wrong cannot hide a subject that is plainly moving. The other way round it does
    reopen: the gate blocks a person who stands still, and p >= LOW sends them to the model.
    The high band skips the model only for classes the detector names reliably.
    """
    if p is None:
        return "scenery" if gate_scenery else "ask"
    if p >= HIGH and cls in SURE_CLASSES:
        return "sure"
    if gate_scenery and p < LOW:
        return "scenery"
    return "ask"
