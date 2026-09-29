#!/usr/bin/env python3
"""Motion in, decision out. Polls motion sensors, asks a local VLM what it sees, decides.

MVP of the production bridge. It keeps the parts that carry the idea, the prompt contract, the
parse, and failing open, and leaves out escalation, shadow comparison and plate collection.
"""
import base64
import datetime
import json
import os
import sys
import time
import urllib.request

import config

OLLAMA = config.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = config.get("VLM_MODEL", "qwen3-vl:4b-instruct")
RING_URL = config.get("RING_URL", "http://127.0.0.1:8099")
HA_URL = config.get("HA_URL", "http://homeassistant.local")
HA_TOKEN = config.get("HA_TOKEN", "")
PUSH_URL = config.get("KUMA_PUSH_URL", "")

POLL_S = 2
EVENT_WINDOW_S = 15
POST_EVENT_S = 8
EVENT_FRAMES = 6
OLLAMA_TIMEOUT = 120

# Which motion sensor maps to which ring channel and human-readable place.
# {"binary_sensor.gate_motion": [4, "Front gate"], ...}
CAMS = json.loads(config.get("CAMS_JSON", "{}"))

# The model cannot know what "on the property" means at your address, so you describe it.
# Everything past the gate is yours; the rest is the street. Rewrite both for your site.
SCENE = config.get("VLM_SCENE", "Frames from a fixed outdoor security camera, day or night. "
                                "A gate and fence separate this private property from a public street.")
RULES = config.get("VLM_RULES", "Judge position by the surface the subject stands or drives on. "
                                "Paths, driveway, garden and steps inside the fence are house. "
                                "The public road is street. The gate is often open and the road is "
                                "usually visible behind it; neither decides it.")
VOCAB = "Vehicles that appear here are cars, motorcycles, bicycles, trucks, vans and buses."

PROMPT = (
    SCENE + " " + VOCAB + " " + RULES + "\n"
    "Step 1. Is there a person or a vehicle in ANY frame? An empty scene is normal and common. "
    "If nothing is there, reply EMPTY and stop.\n"
    "Step 2. If something is there, reply on exactly four lines and nothing else:\n"
    "PERSON: yes or no\n"
    "VEHICLE: the type, or none\n"
    "COLOR: the vehicle's main colour in one word, or none\n"
    "WHERE: house or street"
)

VEHICLES = ("motorcycle", "bicycle", "car", "truck", "van", "bus", "suv", "pickup", "scooter")
ACRONYMS = {"suv"}


def log(*a):
    print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def parse(resp):
    """-> (present, person, vehicle, where).

    A leading EMPTY is only believed when the model names nothing after it. It often answers
    "EMPTY ... PERSON: no VEHICLE: white car", and trusting the first word threw the car away.
    """
    t = " ".join(resp.split()).upper()
    person = "PERSON: YES" in t
    veh = "none"
    if "VEHICLE:" in t:
        field = t.split("VEHICLE:", 1)[1].split("COLOR:")[0].split("WHERE:")[0].split("PERSON:")[0].strip()[:40]
        if field and not field.split()[0].startswith("NONE"):
            veh = next((v for v in VEHICLES if v.upper() in field), field.split()[-1].lower())
    if (t.startswith("EMPTY") or t.startswith("NONE")) and not person and veh == "none":
        return False, False, "none", "none"
    where = "house" if "WHERE: HOUSE" in t else ("street" if "WHERE: STREET" in t else "none")
    return (person or veh != "none"), person, veh, where


def vehicle_color(resp):
    t = " ".join(resp.split()).upper()
    if "COLOR:" not in t:
        return ""
    c = t.split("COLOR:", 1)[1].split("WHERE:")[0].split("PERSON:")[0].split("VEHICLE:")[0].strip()
    c = c.split()[0].lower() if c else ""
    return "" if c in ("", "none", "n/a", "unknown", "na") else c


def title_for(person, veh, where, color=""):
    if veh != "none":
        name = veh.upper() if veh in ACRONYMS else veh.capitalize()
        if color:
            name = f"{color.capitalize()} {name}"
        who = f"{name} with rider" if person else name
    else:
        who = "Person" if person else "Motion"
    return f"{who} ({where})"


def ha(path, method="GET", payload=None):
    req = urllib.request.Request(
        HA_URL + path, method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def frames_for(ch, event_utc):
    """Frames from the ring around the event.

    RTSP pull lags the live scene by a few seconds and the motion event lands after the subject,
    so the window reaches forward as well as back, and we wait before slicing.
    """
    time.sleep(POST_EVENT_S)
    end = event_utc.timestamp() + POST_EVENT_S
    url = f"{RING_URL}/frames?ch={ch}&window={EVENT_WINDOW_S}&n={EVENT_FRAMES}&end={end}"
    try:
        return json.load(urllib.request.urlopen(url, timeout=30)).get("frames") or []
    except Exception as e:
        log(f"  ring error ch{ch}: {type(e).__name__}")
        return []


def ask(path):
    img = base64.b64encode(open(path, "rb").read()).decode()
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "images": [img], "stream": False,
                       "options": {"temperature": 0.1, "num_predict": 100}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT)).get("response", "").strip()
    except Exception as e:
        log(f"  model error on {os.path.basename(path)}: {type(e).__name__}")
        return None


def analyze(frames):
    """Ask frame by frame, stop on the first hit. Returns None when the model is unreachable."""
    reachable = False
    for f in frames:
        resp = ask(f)
        if resp is None:
            continue
        reachable = True
        present, person, veh, where = parse(resp)
        if present:
            return resp, present, person, veh, where
    return ("", False, False, "none", "none") if reachable else None


def handle(ent, event_utc):
    ch, spot = CAMS[ent]
    frames = frames_for(ch, event_utc)
    if not frames:
        log(f"  no frames for {spot}")
        return
    result = analyze(frames)

    if result is None:
        # Fail open. A silent pipeline is worse than a false alarm, because you believe you are covered
        log(f"  {spot}: MODEL UNREACHABLE, escalating without confirmation")
        alarm(spot, "Motion (unconfirmed)", confirmed=False)
        return

    resp, present, person, veh, where = result
    if not present:
        log(f"  {spot}: nothing identified")
        return

    title = title_for(person, veh, where, vehicle_color(resp))
    log(f"  {spot}: {title}")
    if where == "house":
        alarm(spot, title, confirmed=True)


def alarm(spot, title, confirmed):
    """Hook for your own siren, panel or notification. Deliberately not wired here."""
    log(f"  ALARM {spot}: {title} (confirmed={confirmed})")


def heartbeat(_last=[0.0]):
    """Prove this process is alive. A reachable host with a dead bridge otherwise looks healthy."""
    if not PUSH_URL or time.time() - _last[0] < 60:
        return
    _last[0] = time.time()
    try:
        urllib.request.urlopen(PUSH_URL, timeout=5).close()
    except Exception:
        pass


def selfcheck():
    assert parse("PERSON: no VEHICLE: none WHERE: street")[0] is False
    assert parse("PERSON: yes VEHICLE: motorcycle WHERE: street")[1] is True
    assert parse("EMPTY PERSON: no VEHICLE: white car WHERE: street")[0] is True
    assert parse("EMPTY There are no people or vehicles.")[0] is False
    assert vehicle_color("VEHICLE: car COLOR: red WHERE: street") == "red"
    assert vehicle_color("VEHICLE: none COLOR: none WHERE: street") == ""
    assert title_for(False, "car", "street", "red") == "Red Car (street)"
    assert title_for(True, "motorcycle", "street") == "Motorcycle with rider (street)"
    assert title_for(False, "suv", "house") == "SUV (house)"
    print("selfcheck ok")


def main():
    if not CAMS:
        sys.exit("CAMS_JSON is empty, nothing to watch")
    seen = {}
    for s in ha("/api/states"):
        if s["entity_id"] in CAMS:
            seen[s["entity_id"]] = s.get("last_changed")
    log(f"bridge up, watching {len(CAMS)} sensors")
    while True:
        try:
            for s in ha("/api/states"):
                ent = s["entity_id"]
                if ent not in CAMS:
                    continue
                lc = s.get("last_changed")
                if s.get("state") == "on" and lc and lc != seen.get(ent):
                    seen[ent] = lc
                    event = datetime.datetime.fromisoformat(lc).astimezone(datetime.timezone.utc)
                    handle(ent, event)
                elif lc and s.get("state") != "on":
                    seen[ent] = lc
        except Exception as e:
            log("poll error:", e)
        heartbeat()
        time.sleep(POLL_S)


if __name__ == "__main__":
    selfcheck() if len(sys.argv) > 1 and sys.argv[1] == "selfcheck" else main()
