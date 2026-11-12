"""Ask a local VLM what is in one image, and print what the bridge would decide.

    uv run examples/ask.py frame.jpg

Needs Ollama running with a vision model pulled:

    ollama pull qwen3-vl:4b-instruct
"""

import sys
from pathlib import Path

from vlmsec.decide import decide
from vlmsec.vlm import Asker, parse

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    said = Asker().ask(Path(sys.argv[1]))
    if said is None:
        sys.exit("the model could not be reached, is Ollama running?")
    print("--- model said ---")
    print(said or "(no response)")

    a = parse(said)
    print("\n--- parsed ---")
    print(f"  present : {a.present}")
    print(f"  person  : {a.person}")
    print(f"  vehicle : {a.vehicle}")
    print(f"  colour  : {a.color or '-'}")
    print(f"  where   : {a.where}")

    v = decide(a, armed=True)
    print("\n--- decision, if armed ---")
    print(f"  {v.action}: {v.title or 'nothing there'}")
