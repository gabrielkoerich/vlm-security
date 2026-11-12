"""Optional per-camera boundary: a line drawn in pixel space that splits property from street.

The model judges position from what it sees, and it is wrong in a predictable way: through an open
gate it sees the road and calls a car that has already entered "street". A camera does not move,
so the boundary can be drawn once, by hand, on a still frame. Where a line exists it decides;
where none exists the model's WHERE stands.

Config, in GEOMETRY_JSON, one entry per channel:

    {"4": {"a": [0, 520], "b": [1280, 610], "property": "below"}}

`a` and `b` are two points on the boundary in the frame's pixels. `property` says which side of
the line is the property: "below" or "above" for a line across the frame, "left" or "right" for a
line that runs top to bottom. The box's bottom centre, where the subject touches the ground, is
the point tested.

Cost: every camera needs its own line, and a line is only right while the camera does not move.
"""

from dataclasses import dataclass
from typing import Literal

from vlmsec import config, net
from vlmsec.window import Box

type Side = Literal["above", "below", "left", "right"]
type Place = Literal["house", "street", "none"]


@dataclass(frozen=True)
class Line:
    a: tuple[float, float]
    b: tuple[float, float]
    property: Side


def side_of(line: Line, box: Box) -> Place:
    """House when the box's bottom centre is on the property side of the line, else street."""
    cx, cy, _, h = box
    px, py = cx, cy + h / 2
    (ax, ay), (bx, by) = line.a, line.b
    # sign of the cross product, positive is to the left of a->b in image coordinates (y down)
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if line.property in ("below", "above"):
        if ax > bx:
            cross = -cross
        on_property = cross > 0 if line.property == "below" else cross < 0
    else:
        if ay > by:
            cross = -cross
        on_property = cross < 0 if line.property == "right" else cross > 0
    return "house" if on_property else "street"


def _point(raw: object) -> tuple[float, float] | None:
    if not net.is_list(raw) or len(raw) != 2:
        return None
    return net.as_float(raw[0]), net.as_float(raw[1])


SIDES: dict[str, Side] = {"above": "above", "below": "below", "left": "left", "right": "right"}


def _side(raw: object) -> Side | None:
    return SIDES.get(raw) if isinstance(raw, str) else None


def load(raw: str) -> dict[int, Line]:
    """GEOMETRY_JSON as lines per channel. A malformed entry is skipped, never guessed."""
    out: dict[int, Line] = {}
    data = net.loads(raw) if raw.strip() else None
    if not net.is_dict(data):
        return out
    for ch, entry in data.items():
        a, b = _point(net.field(entry, "a")), _point(net.field(entry, "b"))
        side = _side(net.field(entry, "property"))
        if a is not None and b is not None and side is not None and ch.isdigit():
            out[int(ch)] = Line(a, b, side)
    return out


LINES = load(config.get("GEOMETRY_JSON", ""))


def place(ch: int, box: Box | None, model_where: str, lines: dict[int, Line] = LINES) -> str:
    """The line's answer when the camera has one and there is a box, else the model's."""
    line = lines.get(ch)
    if line is None or box is None:
        return model_where
    return side_of(line, box)
