"""Train the hybrid from labelled detections and write the bundle hybrid.py loads.

Labels come from two places, one JSONL row per detection:

  * the bridge's own decisions log (decisions.jsonl): every event the model was asked about
    carries its verdict, so labelling costs no extra model calls
  * human reviews, the same fields: these win over the model's label for the same event

Row format:

    {"frame": "/path/f.jpg", "box": [cx, cy, w, h], "cls": "person", "conf": 0.81,
     "moved": 140.0, "camera": 2, "ts": "2026-01-01T21:14:03", "label": "real"}

`label` is "real" or "false". Rows labelled anything else ("unusable") are dropped. An event is
keyed by (camera, ts), and the first file to label it wins, so pass the reviews first.

Teaching the hybrid from the model's verdicts means it learns the model's mistakes too. That is
why reviews win, and why the sure band skips the model only on the real side.

    uv run --group vision python -m vlmsec.train reviews.jsonl ~/.vlmsec/decisions.jsonl [--out PATH]
"""

import datetime
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vlmsec import hybrid, net
from vlmsec.hybrid import Bundle, Embedder, Event
from vlmsec.window import Box

SEEDS = 3
TREES = 300


@dataclass(frozen=True)
class Label:
    frame: Path
    event: Event
    real: bool
    ts: str


def _box(raw: object) -> Box | None:
    if not net.is_list(raw) or len(raw) < 4:
        return None
    x, y, w, h = (net.as_float(v) for v in raw[:4])
    return x, y, w, h


def parse_row(row: object) -> Label | None:
    label = net.field(row, "label")
    box = _box(net.field(row, "box"))
    ts = net.as_str(net.field(row, "ts"))
    if label not in ("real", "false") or box is None or not ts:
        return None
    try:
        hour = datetime.datetime.fromisoformat(ts).hour
    except ValueError:
        return None
    camera = net.field(row, "camera")
    event = Event(camera=camera if isinstance(camera, int) else -1,
                  cls=net.as_str(net.field(row, "cls")), conf=net.as_float(net.field(row, "conf")),
                  box=box, moved=net.as_float(net.field(row, "moved")), hour=hour)
    return Label(Path(net.as_str(net.field(row, "frame"))), event, label == "real", ts)


def read_labels(files: Iterable[Path]) -> list[Label]:
    """Every usable row, the first file to label an event winning."""
    out: list[Label] = []
    seen: set[tuple[int, str]] = set()
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            got = parse_row(net.loads(line))
            if got is None or (got.event.camera, got.ts) in seen:
                continue
            seen.add((got.event.camera, got.ts))
            out.append(got)
    return out


def fit(labels: Sequence[Label], embed: Embedder, embedder_name: str) -> Bundle:
    ys = np.array(["real" if lb.real else "false" for lb in labels])
    if len({lb.real for lb in labels}) < 2:
        raise ValueError("training needs both real and false labels")
    cameras = sorted({lb.event.camera for lb in labels})
    x_box = [hybrid.features(lb.event, cameras) for lb in labels]
    # class_weight balanced: real events are rare, and a miss costs more than a wasted look
    boxes = [RandomForestClassifier(TREES, class_weight="balanced", random_state=s).fit(x_box, ys)
             for s in range(SEEDS)]
    crops: list[Image.Image] = []
    for lb in labels:
        with Image.open(lb.frame) as img:
            crops.append(hybrid.crop(img.convert("RGB"), lb.event.box))
    emb = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    emb = emb.fit(embed(crops), ys)
    return Bundle(box=boxes, emb=emb, cameras=cameras, embedder=embedder_name,
                  sklearn=sklearn.__version__, labels=len(labels))


def main(argv: Sequence[str]) -> None:
    from vlmsec.embed import MODEL, Dinov2

    args = list(argv)
    out = hybrid.BUNDLE
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    labels = read_labels(Path(a) for a in args)
    bundle = fit(labels, Dinov2(), MODEL)
    hybrid.save(bundle, out)
    real = sum(lb.real for lb in labels)
    print(f"{len(labels)} labels ({real} real), scikit-learn {bundle.sklearn} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
