#!/usr/bin/env python3
"""Ask a local VLM what is in one image, and print what the bridge would decide.

    python3 examples/ask.py frame.jpg

Needs Ollama running with a vision model pulled:

    ollama pull qwen3-vl:4b-instruct
"""
import base64
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from bridge import PROMPT, parse, title_for, vehicle_color  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl:4b-instruct")


def ask(path):
    img = base64.b64encode(open(path, "rb").read()).decode()
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 100},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120)).get("response", "").strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    answer = ask(sys.argv[1])
    print("--- model said ---")
    print(answer or "(no response)")

    present, person, vehicle, where = parse(answer)
    print("\n--- parsed ---")
    print(f"  present : {present}")
    print(f"  person  : {person}")
    print(f"  vehicle : {vehicle}")
    print(f"  colour  : {vehicle_color(answer) or '-'}")
    print(f"  where   : {where}")

    print("\n--- decision ---")
    if not present:
        print("  nothing there, no alarm")
    elif where == "house":
        print(f"  ALARM: {title_for(person, vehicle, where, vehicle_color(answer))}")
    else:
        print(f"  log only: {title_for(person, vehicle, where, vehicle_color(answer))}")
