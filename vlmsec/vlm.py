"""The question to the vision language model, the answer contract, and how the answer is read.

The model runs locally under Ollama. It gets one frame and a short prompt that ends in a fixed
answer format, and the bridge parses that format with plain string matching. A small model
follows a line format far more reliably than it produces valid JSON.

Two lessons are built into the contract:

  * The person and the vehicle get their own WHERE. A person standing inside the gate with a car
    on the road behind them got a single "street" when there was one WHERE for the whole picture.
  * A leading EMPTY is only believed when nothing is named after it. Models often answer
    "EMPTY ... VEHICLE: white car", and trusting the first word threw the car away.
"""

import base64
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vlmsec import config, net

OLLAMA = config.get("OLLAMA_GENERATE_URL", config.OLLAMA_URL.rstrip("/") + "/api/generate")
MODEL = config.get("VLM_MODEL", "qwen3-vl:4b-instruct")
TIMEOUT_S = config.get_float("VLM_TIMEOUT_S", 60.0)
# An HTTP error that comes back fast is usually a runner being swapped (see model_reload.py), so
# it is asked once more. A refused connection or a timeout means Ollama is down, no retry
FAST_ERROR_S = 10.0
RETRY_S = 5.0
RETRY_PAUSE_S = 60.0

# The model cannot know what "on the property" means at your address, so you describe it
SCENE = config.get("VLM_SCENE", ("Frames from a fixed outdoor security camera, day or night. "
                                "A gate and fence separate this private property from a public street."))
RULES = config.get("VLM_RULES", ("Judge position by the surface the subject stands or drives on. "
                                "Paths, driveway, garden and steps inside the fence are house. "
                                "The public road is street. The gate is often open and the road is "
                                "usually visible behind it; neither decides it."))
VOCAB = "Vehicles that appear here are cars, motorcycles, bicycles, trucks, vans and buses."
STEPS = (
    "Step 1. Is there a person or a vehicle in ANY frame? An empty scene is normal and common. "
    "If nothing is there, reply EMPTY and stop.\n"
    "Step 2. Judge the person and the vehicle separately, they are often in different places. "
    "A person sitting inside a vehicle is wherever that vehicle is.\n"
    "Reply on exactly five lines and nothing else:\n"
    "PERSON: yes or no\n"
    "PERSON_WHERE: house or street or none\n"
    "VEHICLE: the type, or none\n"
    "COLOR: the vehicle's main colour in one word, or none\n"
    "VEHICLE_WHERE: house or street or none"
)
PROMPT = f"{SCENE} {VOCAB} {RULES}\n{STEPS}"

VEHICLES = ("motorcycle", "bicycle", "car", "truck", "van", "bus", "suv", "pickup", "scooter")
ACRONYMS = {"suv"}


@dataclass(frozen=True)
class Answer:
    present: bool
    person: bool
    vehicle: str
    where: str
    color: str = ""


NOTHING = Answer(False, False, "none", "none")


def _pos(text: str, key: str) -> str:
    # the lookbehind stops WHERE from matching inside PERSON_WHERE
    m = re.search(rf"(?<![A-Z_]){key}:\s*(HOUSE|STREET)", text)
    return m.group(1).lower() if m else "none"


def _field(text: str, key: str) -> str:
    """The words after `key:` up to the next known key."""
    m = re.search(rf"(?<![A-Z_]){key}:(.*)", text)
    if not m:
        return ""
    return re.split(r"(?<![A-Z_])(?:PERSON|PERSON_WHERE|VEHICLE|COLOR|VEHICLE_WHERE|WHERE):",
                    m.group(1))[0].strip()


def vehicle_color(resp: str) -> str:
    """The COLOR field as one lowercase word, or "" when none. Cosmetic, it feeds the title."""
    words = _field(" ".join(resp.split()).upper(), "COLOR").split()
    c = words[0].lower() if words else ""
    return "" if c in ("", "none", "n/a", "unknown", "na") else c


def parse(resp: str) -> Answer:
    """Read the four or five line answer. Accepts the older single WHERE line too.

    Anything in the VEHICLE field other than "none" is a vehicle, whatever word was used, because
    matching a fixed list missed answers such as "blue SUV". When a person was seen, the person's
    position decides, since a person on the property is what the alarm is for.
    """
    t = " ".join(resp.split()).upper()
    person = "PERSON: YES" in t
    veh = "none"
    raw = _field(t, "VEHICLE")[:40]
    if raw and not raw.split()[0].startswith("NONE"):
        veh = next((v for v in VEHICLES if v.upper() in raw), raw.split()[-1].lower())
    if (t.startswith("EMPTY") or t.startswith("NONE")) and not person and veh == "none":
        return NOTHING
    person_where, veh_where = _pos(t, "PERSON_WHERE"), _pos(t, "VEHICLE_WHERE")
    if person and person_where != "none":
        where = person_where
    elif veh != "none" and veh_where != "none":
        where = veh_where
    else:
        where = _pos(t, "WHERE")
    return Answer(person or veh != "none", person, veh, where, vehicle_color(resp))


def title_for(a: Answer) -> str:
    if a.vehicle != "none":
        name = a.vehicle.upper() if a.vehicle in ACRONYMS else a.vehicle.capitalize()
        if a.color:
            name = f"{a.color.capitalize()} {name}"
        who = f"{name} with rider" if a.person else name
    else:
        who = "Person" if a.person else "Motion"
    return f"{who} ({a.where})"


def body_for(image: Path, prompt: str = PROMPT, model: str = MODEL) -> bytes:
    # keep_alive -1 pins the model: Ollama's default unloads it after 5 idle minutes, and the first
    # event after a quiet spell would then wait for a cold load
    return json.dumps({
        "model": model, "prompt": prompt, "stream": False, "keep_alive": -1,
        "images": [base64.b64encode(image.read_bytes()).decode()],
        "options": {"temperature": 0.1, "num_predict": 100},
    }).encode()


class Asker:
    """Asks the model about one frame. Returns None when the model could not be reached.

    Retry rule: one retry after RETRY_S, only for an HTTP error that came back within
    FAST_ERROR_S, which is what a runner between an unload and a load looks like. A refused
    connection or a timeout is a dead Ollama and fails at once. After a retry that failed too,
    no retry is tried for RETRY_PAUSE_S, so a down model cannot add RETRY_S to every event.
    """

    def __init__(self, url: str = OLLAMA, timeout: float = TIMEOUT_S,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.url: str = url
        self.timeout: float = timeout
        self.clock: Callable[[], float] = clock
        self.sleep: Callable[[float], None] = sleep
        self.retry_failed_at: float | None = None

    def _once(self, body: bytes) -> str:
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        return net.as_str(net.field(net.loads(net.fetch(req, self.timeout)), "response")).strip()

    def _may_retry(self, err: Exception, took: float) -> bool:
        paused = (self.retry_failed_at is not None
                  and self.clock() - self.retry_failed_at < RETRY_PAUSE_S)
        return isinstance(err, urllib.error.HTTPError) and took <= FAST_ERROR_S and not paused

    def ask(self, image: Path, prompt: str = PROMPT) -> str | None:
        body = body_for(image, prompt)
        start = self.clock()
        try:
            return self._once(body)
        except (OSError, ValueError) as err:
            if not self._may_retry(err, self.clock() - start):
                return None
        self.sleep(RETRY_S)
        try:
            return self._once(body)
        except (OSError, ValueError):
            self.retry_failed_at = self.clock()
            return None
