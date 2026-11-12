"""Motion in, decision out. Polls the home automation hub for motion, judges it, acts back through it.

One event, in order, cheapest first:

  1. wait AFTER_S, then read the ring frames from BEFORE_S before the event to AFTER_S after it
  2. send every DETECT_STRIDE-th frame to the detector in one request (window.detect)
  3. reduce to one subject: top confidence, best view, how far it moved
  4. scenery gate: a subject that moved under MOVE_MIN is scenery
  5. hybrid score, when a bundle exists: confirm the gate, reopen it, or skip the model (hybrid.route)
  6. ask the VLM about the best view, unless steps 4 and 5 settled it
  7. place the subject: a geometry line if the camera has one, else the model's WHERE
  8. decide: on the property and armed raises the alarm, see decide.py

Every judged event is appended to DECISIONS as one JSON row. Those rows are the training labels
for the hybrid (train.py), so the hybrid learns from the model's own decisions.

    python -m vlmsec.bridge
"""

import datetime
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from vlmsec import config, decide, frame_ring, geometry, hybrid, net, window
from vlmsec.model_reload import BUSY_FILE
from vlmsec.vlm import NOTHING, Answer, Asker, parse

POLL_S = 2.0
FRAME_W = config.get_float("FRAME_W", float(frame_ring.MAX_SIDE))
DETECTOR_URL = config.get("DETECTOR_URL", "http://127.0.0.1:8098")
# A detection this confident stands even when the model says nothing is there
DETECTOR_TRUST = 0.85
# Without the detector, this many frames spread over the window go to the model
FALLBACK_FRAMES = 6
DECISIONS = config.RUN_DIR / "decisions.jsonl"
HA_URL = config.HA_URL.rstrip("/")
ARMED_ENTITY = config.get("ARMED_ENTITY", "alarm_control_panel.home")
ALARM_SERVICE = config.get("ALARM_SERVICE", "script/turn_on")
ALARM_TARGET = config.get("ALARM_TARGET", "script.siren")
NOTIFY_SERVICE = config.get("NOTIFY_SERVICE", "notify/notify")

type Kind = Literal["empty", "scenery", "sure", "asked", "unreachable"]
type AskFn = Callable[[Path], str | None]


@dataclass(frozen=True)
class Camera:
    ch: int
    spot: str


def cameras(raw: str) -> dict[str, Camera]:
    """CAMS_JSON: {"binary_sensor.gate_motion": [4, "Front gate"]}, motion entity to channel and name."""
    data = net.loads(raw) if raw.strip() else None
    out: dict[str, Camera] = {}
    if not net.is_dict(data):
        return out
    for ent, val in data.items():
        if net.is_list(val) and len(val) == 2 and isinstance(val[0], int):
            out[ent] = Camera(val[0], net.as_str(val[1]))
    return out


@dataclass(frozen=True)
class Outcome:
    kind: Kind
    answer: Answer
    human: bool
    p: float | None = None


def spread(paths: list[str], n: int = FALLBACK_FRAMES) -> list[str]:
    """n frames evenly over the window. Used only when the detector is down."""
    if len(paths) <= n:
        return paths
    return [paths[round(i * (len(paths) - 1) / (n - 1))] for i in range(n)]


def sure_answer(cls: str) -> Answer:
    """What a sure hybrid call says without the model. A dog counts as a person, it is a body on the property."""
    person = cls in ("person", "dog")
    return Answer(True, person, "none" if person else cls, "none")


def evaluate(ch: int, seen: window.Seen | None, frames: list[str], p: float | None, ask: AskFn,
             lines: dict[int, geometry.Line] = geometry.LINES) -> Outcome:
    """Steps 4 to 7 for one event, with the model and the score passed in so tests can fake them."""
    if seen is None:
        # the detector is down, so the model sees a spread of the window and the event itself
        # stands in for a human detection if the model is down too
        reached = False
        for f in spread(frames):
            resp = ask(Path(f))
            if resp is None:
                continue
            reached = True
            ans = parse(resp)
            if ans.present:
                return Outcome("asked", ans, ans.person)
        return Outcome("asked" if reached else "unreachable", NOTHING, not reached)
    best = seen.best
    if best is None:
        return Outcome("empty", NOTHING, False)
    human = best.cls == "person"
    match hybrid.route(window.is_scenery(seen), p, best.cls):
        case "scenery":
            return Outcome("scenery", NOTHING, False, p)
        case "sure":
            ans = sure_answer(best.cls)
            where = decide.place_sure(geometry.place(ch, best.box, "none", lines))
            return Outcome("sure", Answer(True, ans.person, ans.vehicle, where), human, p)
        case "ask":
            resp = ask(Path(best.path))
            if resp is None:
                return Outcome("unreachable", NOTHING, human, p)
            ans = parse(resp)
            if not ans.present and best.conf >= DETECTOR_TRUST:
                ans = Answer(True, human, "none" if human else best.cls, ans.where)
            where = geometry.place(ch, best.box, ans.where, lines) if ans.present else ans.where
            return Outcome("asked", Answer(ans.present, ans.person, ans.vehicle, where, ans.color), human, p)


class Hub:
    """The home automation hub: motion source, armed state, and where alarms and notices go."""

    def __init__(self, url: str = HA_URL, token: str = "") -> None:
        self.url: str = url
        self.headers: dict[str, str] = {"Authorization": f"Bearer {token}"}

    def states(self) -> list[object]:
        got = net.get_json(f"{self.url}/api/states", 15, self.headers)
        return got if net.is_list(got) else []

    def armed(self) -> bool:
        """Anything but disarmed counts as armed, an unreachable hub included."""
        try:
            got = net.get_json(f"{self.url}/api/states/{ARMED_ENTITY}", 10, self.headers)
        except (OSError, ValueError):
            return True
        return net.field(got, "state") != "disarmed"

    def call(self, service: str, data: dict[str, object]) -> None:
        try:
            net.post_json(f"{self.url}/api/services/{service}", data, 30, self.headers)
        except (OSError, ValueError) as e:
            log(f"  {service} failed: {type(e).__name__}")


def log(*parts: object) -> None:
    print(datetime.datetime.now().strftime("%H:%M:%S"), *parts, flush=True)


def act(hub: Hub, action: decide.Action, spot: str, title: str) -> None:
    if action in ("ignore", "log"):
        return
    if action == "alarm":
        hub.call(ALARM_SERVICE, {"entity_id": ALARM_TARGET})
    urgent = action in ("alarm", "alert")
    hub.call(NOTIFY_SERVICE, {"title": f"{title} at {spot}" if not urgent else f"ALARM: {title} at {spot}",
                              "message": f"{action}: {title}, {spot}, {time.strftime('%H:%M')}"})


def record(cam: Camera, event: datetime.datetime, seen: window.Seen | None, out: Outcome) -> None:
    """One row per judged event, the hybrid's training data. A model verdict carries a label."""
    best = seen.best if seen else None
    row: dict[str, object] = {"ts": event.astimezone().isoformat(timespec="seconds"), "camera": cam.ch,
                              "kind": out.kind, "p": out.p, "answer": asdict(out.answer)}
    if best:
        row.update(frame=best.path, box=list(best.box), cls=best.cls, conf=best.conf,
                   moved=seen.moved if seen else 0.0)
    if out.kind == "asked" and best:
        row["label"] = "real" if out.answer.present else "false"
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("a") as f:
        f.write(json.dumps(row) + "\n")


class Bridge:
    def __init__(self, hub: Hub, cams: dict[str, Camera]) -> None:
        self.hub: Hub = hub
        self.cams: dict[str, Camera] = cams
        self.asker: Asker = Asker()
        self.vision_down: decide.VisionDown = decide.VisionDown()
        self.model: hybrid.Hybrid | None = None
        bundle = hybrid.load()
        if bundle is not None:
            from vlmsec.embed import Dinov2

            self.model = hybrid.Hybrid(bundle, Dinov2(bundle.embedder))

    def ask(self, image: Path) -> str | None:
        # model_reload.py reads this marker and never recycles a runner while it is fresh
        BUSY_FILE.parent.mkdir(parents=True, exist_ok=True)
        BUSY_FILE.write_text(str(time.time()))
        try:
            return self.asker.ask(image)
        finally:
            BUSY_FILE.unlink(missing_ok=True)

    def score(self, cam: Camera, seen: window.Seen | None, event: datetime.datetime) -> float | None:
        best = seen.best if seen else None
        if self.model is None or seen is None or best is None:
            return None
        e = hybrid.Event(cam.ch, best.cls, best.conf, best.box, seen.moved, event.astimezone().hour)
        try:
            return self.model.score(Path(best.path), e)
        except (OSError, ValueError) as err:
            log(f"  hybrid unavailable ({type(err).__name__}), using the gate alone")
            return None

    def handle(self, cam: Camera, event: datetime.datetime) -> None:
        time.sleep(window.AFTER_S)
        end = event.timestamp() + window.AFTER_S
        frames = frame_ring.frames_in_window(cam.ch, window.BEFORE_S + window.AFTER_S, end)
        if not frames:
            log(f"  {cam.spot}: no frames in the ring")
            return
        seen = window.detect(DETECTOR_URL, frames, FRAME_W)
        out = evaluate(cam.ch, seen, frames, self.score(cam, seen, event), self.ask)
        armed = self.hub.armed()
        if out.kind == "unreachable":
            action = self.vision_down.on_human(time.time(), armed) if out.human else "log"
            log(f"  {cam.spot}: MODEL UNREACHABLE, human={out.human}, {action}")
            act(self.hub, action, cam.spot, "Person (unconfirmed)")
        else:
            verdict = decide.decide(out.answer, armed)
            log(f"  {cam.spot}: {out.kind}, p={out.p}, {verdict.action} {verdict.title}")
            act(self.hub, verdict.action, cam.spot, verdict.title)
        record(cam, event, seen, out)

    def run(self) -> None:
        seen: dict[str, str] = {}
        log(f"bridge up, watching {len(self.cams)} sensors, hybrid {'on' if self.model else 'off'}")
        while True:
            try:
                for s in self.hub.states():
                    ent = net.as_str(net.field(s, "entity_id"))
                    changed = net.as_str(net.field(s, "last_changed"))
                    cam = self.cams.get(ent)
                    if cam is None or not changed or changed == seen.get(ent):
                        continue
                    first = ent not in seen
                    seen[ent] = changed
                    if net.field(s, "state") == "on" and not first:
                        log(f"event {cam.spot} ch{cam.ch}")
                        self.handle(cam, datetime.datetime.fromisoformat(changed))
            except (OSError, ValueError) as e:
                log(f"poll error: {type(e).__name__}: {e}")
            time.sleep(POLL_S)


def main() -> None:
    cams = cameras(config.get("CAMS_JSON", ""))
    if not cams:
        sys.exit("CAMS_JSON is empty, nothing to watch")
    Bridge(Hub(token=config.ha_token(allow_passbox=False)), cams).run()


if __name__ == "__main__":
    main()
