import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from vlmsec import hybrid, train
from vlmsec.hybrid import BundleMismatch, Event, Hybrid, crop, features, route


def mean_colour(images: Sequence[Image.Image]) -> NDArray[np.float32]:
    """A stand-in embedder: the mean RGB of each crop."""
    return np.array([np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0) for img in images],
                    dtype=np.float32)


def test_bands_without_a_model_follow_the_gate() -> None:
    assert route(True, None, "person") == "scenery"
    assert route(False, None, "person") == "ask"


def test_bands_with_a_model() -> None:
    assert route(True, 0.05, "person") == "scenery"
    assert route(True, 0.3, "person") == "ask", "the hybrid reopens what the gate blocked"
    assert route(False, 0.05, "person") == "ask", "the low band never closes what the gate let through"
    assert route(False, 0.95, "car") == "sure"
    assert route(True, 0.95, "dog") == "sure"
    assert route(False, 0.95, "umbrella") == "ask", "only reliable classes skip the model"


def test_features_one_hot() -> None:
    e = Event(camera=2, cls="car", conf=0.8, box=(100.0, 50.0, 20.0, 10.0), moved=5.0, hour=21)
    f = features(e, [1, 2, 3])
    assert f[:8] == [0.8, 20.0, 10.0, 200.0, 100.0, 50.0, 5.0, 21.0]
    assert f[8:11] == [0.0, 1.0, 0.0]
    assert sum(f[11:]) == 1.0


def test_crop_is_square_and_clamped() -> None:
    img = Image.new("RGB", (400, 300))
    assert crop(img, (200.0, 150.0, 40.0, 80.0)).size == (120, 120)
    assert crop(img, (390.0, 290.0, 40.0, 80.0)).size == (70, 70)


def write_labels(tmp: Path) -> Path:
    """Synthetic frames: real subjects are bright and move, scenery is dark and still."""
    rows: list[str] = []
    for i in range(40):
        real = i % 2 == 0
        frame = tmp / f"{i}.png"
        Image.new("RGB", (320, 240), (220, 220, 220) if real else (20, 40, 20)).save(frame)
        rows.append(json.dumps({
            "frame": str(frame), "box": [160, 120, 40 if real else 20, 80 if real else 20],
            "cls": "person", "conf": 0.8 if real else 0.4, "moved": 200.0 if real else 10.0,
            "camera": 1 + i % 3, "ts": f"2026-01-{1 + i % 28:02d}T{i % 24:02d}:00:00",
            "label": "real" if real else "false"}))
    path = tmp / "labels.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_reviews_win_over_model_labels(tmp_path: Path) -> None:
    row = {"frame": "f.jpg", "box": [1, 2, 3, 4], "cls": "person", "conf": 0.5, "moved": 0,
           "camera": 1, "ts": "2026-01-01T10:00:00"}
    review = tmp_path / "review.jsonl"
    review.write_text(json.dumps({**row, "label": "real"}) + "\n")
    model = tmp_path / "model.jsonl"
    model.write_text(json.dumps({**row, "label": "false"}) + "\n"
                     + json.dumps({**row, "ts": "2026-01-01T11:00:00", "label": "unusable"}) + "\n")
    got = train.read_labels([review, model])
    assert len(got) == 1
    assert got[0].real


def test_train_save_load_and_score(tmp_path: Path) -> None:
    labels = train.read_labels([write_labels(tmp_path)])
    bundle = train.fit(labels, mean_colour, "mean-colour")
    out = tmp_path / "hybrid.joblib"
    hybrid.save(bundle, out)
    loaded = hybrid.load(out)
    assert loaded is not None
    assert loaded.labels == 40
    h = Hybrid(loaded, mean_colour)
    real = Event(1, "person", 0.8, (160.0, 120.0, 40.0, 80.0), 200.0, 3)
    still = Event(1, "person", 0.4, (160.0, 120.0, 20.0, 20.0), 10.0, 3)
    assert h.score(Path(labels[0].frame), real) > 0.9
    assert h.score(Path(labels[1].frame), still) < 0.1


def test_load_refuses_another_sklearn(tmp_path: Path) -> None:
    labels = train.read_labels([write_labels(tmp_path)])
    bundle = train.fit(labels, mean_colour, "mean-colour")
    bundle.sklearn = "0.0.1"
    out = tmp_path / "old.joblib"
    hybrid.save(bundle, out)
    with pytest.raises(BundleMismatch):
        hybrid.load(out)


def test_no_bundle_is_none(tmp_path: Path) -> None:
    assert hybrid.load(tmp_path / "missing.joblib") is None


def test_training_needs_both_labels(tmp_path: Path) -> None:
    labels = [lb for lb in train.read_labels([write_labels(tmp_path)]) if lb.real]
    with pytest.raises(ValueError):
        train.fit(labels, mean_colour, "mean-colour")
