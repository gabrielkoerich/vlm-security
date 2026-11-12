"""Memory guard for pinned Ollama runners. Run it every few minutes from cron or launchd.

The bridge pins the model with keep_alive -1, so the first event after a quiet hour does not wait
for a cold load. A pinned runner never unloads, and over days its resident size grows until the
machine swaps, and in the worst case runs out of memory with the alarm depending on it. So every
run measures each loaded model's runner and reloads the ones over MAX_RUNNER_MB.

Three rules keep the guard from doing harm:

  * Never while the bridge is asking the model about an event (the busy file, see `busy`).
  * After a reload, check that a NEW runner started (see `after_reload`). Ollama keeps a runner
    that is serving a request, so an unload can return while the old process is still up.
  * A model whose fresh runner is already over the limit starts that big. Reloading it again
    would loop forever, so it is left alone until it is seen under the limit (see `may_trim`).

Runner size here is resident set size from `ps`, which works on Linux and macOS. On macOS the
physical footprint (proc_pid_rusage through libproc, the number Activity Monitor shows) also
counts GPU memory and is the better measure, it is not ported here to keep this portable.

    python -m vlmsec.model_reload
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from vlmsec import config, net

OLLAMA = config.OLLAMA_URL.rstrip("/")
MAX_RUNNER_MB = config.get_int("MAX_RUNNER_MB", 7168)
BUSY_FILE = config.RUN_DIR / "bridge.busy"
BUSY_STALE_S = 90.0
STATE = config.RUN_DIR / "model-reload.json"

type After = Literal["same", "small", "big"]


def busy(path: Path = BUSY_FILE, now: float | None = None, stale_s: float = BUSY_STALE_S) -> bool:
    """True while the bridge is asking the model. A marker older than stale_s is a crashed bridge."""
    try:
        stamp = float(path.read_text())
    except (OSError, ValueError):
        return False
    return (time.time() if now is None else now) - stamp < stale_s


def judge(sizes: Mapping[str, Sequence[int | None]], files: Mapping[str, str],
          limit: int = MAX_RUNNER_MB) -> tuple[list[str], list[str]]:
    """-> (models over the limit, models not measured). sizes maps model file to runner MB.

    A model whose runner cannot be found or measured is never taken as small.
    """
    over: list[str] = []
    blind: list[str] = []
    for model, path in files.items():
        mbs = [mb for mb in sizes.get(path, []) if mb is not None]
        if not mbs:
            blind.append(model)
        elif max(mbs) > limit:
            over.append(model)
    return over, blind


def after_reload(old: Sequence[int], new: Mapping[int, int | None], limit: int = MAX_RUNNER_MB) -> After:
    """Runner pids before the reload, and pid -> MB after it.

    "same" when no new runner started or an old one is still over the limit, so try again next
    run. "big" when the new runner is already over the limit. "small" otherwise.
    """
    if not new or set(new) <= set(old):
        return "same"
    if any(mb is not None and mb > limit for pid, mb in new.items() if pid in old):
        return "same"
    fresh = [mb for pid, mb in new.items() if pid not in old and mb is not None]
    return "big" if fresh and max(fresh) > limit else "small"


def may_trim(seen_small: Mapping[str, bool], model: str) -> bool:
    """A model never tracked, or seen under the limit since its last reload, may be reloaded."""
    return seen_small.get(model, True)


def runners(ps_output: str) -> dict[str, dict[int, int]]:
    """Model file -> {pid: RSS in MB}, from `ps -Ao pid=,rss=,command=`.

    Ollama starts one runner process per loaded model with the model file in a --model argument.
    """
    found: dict[str, dict[int, int]] = {}
    for line in ps_output.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, rss_kb, cmd = int(parts[0]), int(parts[1]), parts[2]
        if "runner" not in cmd and "llama-server" not in cmd:
            continue
        for flag in (" --model ", " --model=", " -m "):
            if flag in cmd:
                path = cmd.split(flag, 1)[1].split(" -", 1)[0].strip()
                found.setdefault(path, {})[pid] = rss_kb // 1024
                break
    return found


def ps() -> str:
    return subprocess.run(["ps", "-Ao", "pid=,rss=,command="], capture_output=True, text=True,
                          check=False).stdout


def loaded() -> list[str]:
    models = net.field(net.get_json(f"{OLLAMA}/api/ps", 10), "models")
    return [net.as_str(net.field(m, "name")) for m in (models if net.is_list(models) else [])]


def model_file(name: str) -> str | None:
    """The blob path Ollama runs for this model, read from its modelfile's FROM line."""
    show = net.post_json(f"{OLLAMA}/api/show", {"model": name}, 30)
    for line in net.as_str(net.field(show, "modelfile")).splitlines():
        if line.startswith("FROM /"):
            return line[5:].strip()
    return None


def reload(model: str) -> None:
    """Unload, wait for Ollama to drop it, then pin it again so the next event finds it warm."""
    net.post_json(f"{OLLAMA}/api/generate", {"model": model, "keep_alive": 0}, 300)
    for _ in range(20):
        if model not in loaded():
            break
        time.sleep(1)
    net.post_json(f"{OLLAMA}/api/generate", {"model": model, "keep_alive": -1}, 300)


def load_state(path: Path = STATE) -> dict[str, bool]:
    # an unreadable file counts as all seen small, so a bad file can never switch the guard off
    try:
        raw = net.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, bool)} if net.is_dict(raw) else {}


def save_state(seen: Mapping[str, bool], path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(seen)))
    os.replace(tmp, path)


def run() -> bool:
    """One pass. Returns True when something needs a human: a blind runner or one that starts big."""
    files = {m: f for m in loaded() if (f := model_file(m))}
    found = runners(ps())
    over, blind = judge({f: list(p.values()) for f, p in found.items()}, files)
    bad = bool(blind)
    for m in blind:
        print(f"{m}: runner not found or not measured, its size is not watched")
    seen = load_state()
    for m in files:
        if m not in over and m not in blind:
            seen[m] = True
    todo = [m for m in over if may_trim(seen, m)]
    for m in set(over) - set(todo):
        print(f"{m}: still over {MAX_RUNNER_MB} MB after its last reload, it starts that big")
        bad = True
    if todo and busy():
        print(f"{', '.join(todo)}: the bridge is asking the model, reload left to the next run")
        save_state(seen)
        return bad
    for m in todo:
        before = list(found.get(files[m], {}))
        print(f"{m}: runner over {MAX_RUNNER_MB} MB, reloading")
        reload(m)
        got = after_reload(before, runners(ps()).get(files[m], {}))
        if got == "same":
            print(f"{m}: no new runner started, trying again next run")
            bad = True
        elif got == "big":
            print(f"{m}: the new runner already starts over {MAX_RUNNER_MB} MB, not reloading it again")
            seen[m] = False
            bad = True
        else:
            seen[m] = True
    save_state(seen)
    return bad


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
