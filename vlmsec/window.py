"""From a motion event to one detection: the event window, the detector pass and the scenery gate.

A motion event says that something changed, not what or where. The frames around it already sit
in the ring (frame_ring.py), so this module reads the window, sends a sample of it to the object
detector in one request, and reduces the answer to a single `Detection`:

  * the decision keeps the top confidence of the whole window
  * the picture is the clearest view of that class (see `best_view`)
  * `moved` says how far that subject travelled, and a subject that never travels is scenery

All of it is arithmetic on boxes. No model is asked anything here, which is why it runs first.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from vlmsec import config, net

# The event lands after the subject (stream lag plus the camera's own delay), so the window
# reaches back further than forward. The bridge waits AFTER_S before reading it
BEFORE_S = config.get_float("EVENT_BEFORE_S", 15.0)
AFTER_S = config.get_float("EVENT_AFTER_S", 8.0)
DETECT_STRIDE = config.get_int("DETECT_STRIDE", 4)
DETECTOR_CONF = 0.25
EDGE = 0.02
MOVE_MIN = config.get_float("MOVE_MIN", 100.0)
CLUSTER_PX = 180.0

type Box = tuple[float, float, float, float]
"""Centre x, centre y, width, height in pixels, the detector's xywh"""


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: Box
    path: str


@dataclass(frozen=True)
class Seen:
    """The window reduced to one subject, or nothing when `best` is None."""

    best: Detection | None
    moved: float
    frames: int


def sample_frames(paths: Sequence[str], stride: int = DETECT_STRIDE) -> list[str]:
    """Every `stride`-th frame, plus the last one whatever the stride lands on.

    Reading every frame is most of an event's latency. Replaying real events at strides 1, 2, 4
    and 8 gave the same class each time with confidence within a few hundredths, and a subject
    has to be in shot for under a few frames before stride 4 expects to miss it. The last frame
    is kept because a subject arriving at the end of the window is the one most likely to matter.
    """
    if stride <= 1 or len(paths) <= stride:
        return list(paths)
    kept = list(paths[::stride])
    if kept[-1] != paths[-1]:
        kept.append(paths[-1])
    return kept


def best_view(found: Sequence[Detection], frame_w: float) -> Detection:
    """The top confidence for the decision, and the clearest picture of that class for the frame.

    The most confident box is often the subject entering, cut by the side of the frame. Among the
    boxes of the top class, the largest one that does not touch a side edge (EDGE of the width)
    gives the picture. Known limitation: there is no sharpness check, so a large blurred box can
    still win over a smaller sharp one.
    """
    top = max(found, key=lambda d: d.conf)

    def view(d: Detection) -> tuple[bool, float]:
        cx, _, w, h = d.box
        whole = cx - w / 2 > EDGE * frame_w and cx + w / 2 < (1 - EDGE) * frame_w
        return whole, w * h

    same = [d for d in found if d.cls == top.cls and d.conf >= DETECTOR_CONF]
    pic = max(same or [top], key=view)
    return replace(pic, conf=top.conf)


def moved(boxes: Sequence[Box], anchor: Box, cluster_px: float = CLUSTER_PX) -> float:
    """How far one subject travelled across the window: the largest of x span, y span and height span.

    Only boxes near the chosen one count, otherwise two still objects in different corners read
    as one large movement. Height counts because someone walking straight at the camera barely
    shifts their centre but grows, while a parked object keeps its size.
    """
    near = [b for b in boxes
            if abs(b[0] - anchor[0]) <= cluster_px and abs(b[1] - anchor[1]) <= cluster_px]
    if len(near) < 2:
        return 0.0
    xs, ys, hs = [b[0] for b in near], [b[1] for b in near], [b[3] for b in near]
    return round(max(max(xs) - min(xs), max(ys) - min(ys), max(hs) - min(hs)), 1)


def is_scenery(seen: Seen, move_min: float = MOVE_MIN) -> bool:
    """A subject that never travels is scenery: a parked bin, a plant, a reflection.

    The threshold must sit well above the jitter of a still object's box, which light and wind
    move by tens of pixels. Defaults to 100 px at a 1280 px wide frame.
    """
    return seen.best is not None and seen.frames > 0 and seen.moved < move_min


def summarise(results: Sequence[tuple[str, Sequence[tuple[str, float, Box]]]], frame_w: float) -> Seen:
    """Detector output, per frame `(path, [(cls, conf, box)])`, reduced to one subject."""
    found = [Detection(cls, conf, box, path) for path, dets in results for cls, conf, box in dets]
    if not found:
        return Seen(None, 0.0, len(results))
    best = best_view(found, frame_w)
    travel = moved([d.box for d in found if d.cls == best.cls], best.box)
    return Seen(best, travel, len(results))


def _box(raw: object) -> Box | None:
    if not net.is_list(raw) or len(raw) < 4:
        return None
    x, y, w, h = (net.as_float(v) for v in raw[:4])
    return x, y, w, h


def parse_detector(answer: object) -> list[tuple[str, list[tuple[str, float, Box]]]]:
    """The detector service's JSON, see detector_service.py, as typed tuples."""
    out: list[tuple[str, list[tuple[str, float, Box]]]] = []
    results = net.field(answer, "results")
    for r in results if net.is_list(results) else []:
        dets: list[tuple[str, float, Box]] = []
        raw = net.field(r, "detections")
        for d in raw if net.is_list(raw) else []:
            box = _box(net.field(d, "box"))
            if box is not None:
                dets.append((net.as_str(net.field(d, "cls")), net.as_float(net.field(d, "conf")), box))
        out.append((net.as_str(net.field(r, "path")), dets))
    return out


def detect(url: str, paths: Sequence[str], frame_w: float, timeout: float = 120) -> Seen | None:
    """One request for the whole sampled window. None means the detector never ran.

    "It saw nothing" and "it never looked" lead to different decisions, so they stay apart.
    """
    sample = sample_frames(paths)
    try:
        answer = net.post_json(f"{url}/detect", {"paths": sample, "conf": DETECTOR_CONF}, timeout)
    except (OSError, ValueError):
        return None
    return summarise(parse_detector(answer), frame_w)
