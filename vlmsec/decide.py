"""The decision: what an event's verdict does, given whether the alarm is armed.

On the property and armed raises the alarm. On the property and disarmed notifies. Anything on the
street is logged. Nothing seen is ignored.

Two rules decide what happens when part of the pipeline is missing:

  * Fail open. A vision pipeline that silently stops triggering is worse than none, because you
    believe you are covered. When the model is unreachable and the detector saw a person while
    armed, the event still counts, see `VisionDown`.
  * A sure call with no placement counts as on the property. When the hybrid skips the model and
    no geometry line covers the camera, nothing says where the subject is. Treating that as
    inside costs a false alarm at worst, treating it as outside could cost a missed one.
"""

from dataclasses import dataclass, field
from typing import Literal

from vlmsec.vlm import Answer, title_for

type Action = Literal["alarm", "alert", "notify", "log", "ignore"]

# Two detections closer than MIN_GAP_S are one subject passing, not a second sighting
SECOND_WITHIN_S = 120.0
MIN_GAP_S = 45.0


@dataclass(frozen=True)
class Verdict:
    action: Action
    title: str


def place_sure(where: str) -> str:
    """A sure call that no geometry placed counts as on the property."""
    return "house" if where == "none" else where


def decide(answer: Answer, armed: bool) -> Verdict:
    if not answer.present:
        return Verdict("ignore", "")
    title = title_for(answer)
    if answer.where == "house":
        return Verdict("alarm" if armed else "notify", title)
    return Verdict("log", title)


@dataclass
class VisionDown:
    """Fail open, refined: with the model unreachable, one human detection alerts and a second fires.

    The plain rule (fire on any human detection while the model is down) fires on every shadow
    and every passer-by the detector calls a person, which happens most nights. Requiring a
    second detection within SECOND_WITHIN_S keeps a person who stays, and drops one false box.
    The first detection still sends an urgent alert, so a human sees it at once.
    """

    within_s: float = SECOND_WITHIN_S
    min_gap_s: float = MIN_GAP_S
    _last: float | None = field(default=None, init=False)

    def on_human(self, now: float, armed: bool) -> Action:
        """One human detection while the model is unreachable."""
        if not armed:
            return "notify"
        last = self._last
        if last is not None and now - last < self.min_gap_s:
            return "ignore"
        self._last = now
        if last is not None and now - last <= self.within_s:
            self._last = None
            return "alarm"
        return "alert"
