"""Every host, path and secret this repo needs, resolved at runtime.

Nothing about a specific installation is in the repository. Values resolve in this order:

  1. environment variable
  2. untracked local file, ~/.home_network.conf (HOME_NETWORK_CONF overrides the path)
  3. passbox, only when asked: HN_USE_PASSBOX=1 or secret(..., allow_passbox=True)
  4. a generic default, never a real address

Steps 1, 2 and 4 are all you need. Step 3 is optional and macOS only: passbox
(https://github.com/gabrielkoerich/passbox) gates each secret behind Touch ID.

Long-running daemons must never reach step 3. Under launchd there is no login session to answer
the Touch ID prompt, so the call hangs instead of failing. Daemons read the untracked file.

File format, one KEY=value per line, # starts a comment:

    DVR_HOST=dvr.local
    HA_URL=http://homeassistant.local

    python -m vlmsec.config     show what resolved, never fetches a secret
"""

import os
import subprocess
from pathlib import Path

CONF_PATH = Path(os.environ.get("HOME_NETWORK_CONF", "~/.home_network.conf")).expanduser()
PASSBOX_BOX = os.environ.get("HN_PASSBOX_BOX", "vlmsec/config")


def load(path: Path = CONF_PATH) -> dict[str, str]:
    vals: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return vals
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            vals[key.strip()] = value.strip()
    return vals


_FILE = load()


def get(key: str, default: str = "") -> str:
    """Non-secret config. Never touches passbox, so it is always safe to call."""
    return os.environ.get(key) or _FILE.get(key) or default


def get_float(key: str, default: float) -> float:
    raw = get(key)
    return float(raw) if raw else default


def get_int(key: str, default: int) -> int:
    raw = get(key)
    return int(raw) if raw else default


def require(key: str) -> str:
    """For values with no safe default. Fail loudly rather than pretend."""
    value = get(key)
    if not value:
        raise RuntimeError(f"{key} is not set. Put it in {CONF_PATH} or export it.")
    return value


def passbox_field(field: str, box: str = PASSBOX_BOX, timeout: float = 120) -> str:
    """One field of a passbox entry. Interactive use only, this can raise Touch ID."""
    try:
        done = subprocess.run(["passbox", "get", box, "--field", field],
                              capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = done.stdout.splitlines()
    return lines[0].strip() if done.returncode == 0 and lines else ""


def secret(env_key: str, field: str | None = None, allow_passbox: bool | None = None) -> str:
    """A credential: env, then the untracked file, then passbox only if allowed.

    allow_passbox defaults to HN_USE_PASSBOX=1, so a daemon that never sets it can never hang on
    a Touch ID prompt it has no session to answer.
    """
    value = os.environ.get(env_key) or _FILE.get(env_key)
    if value:
        return value
    if allow_passbox is None:
        allow_passbox = os.environ.get("HN_USE_PASSBOX") == "1"
    if allow_passbox and field:
        return passbox_field(field)
    return ""


def ha_token(allow_passbox: bool | None = None) -> str:
    return secret("HA_TOKEN", field="ha-token", allow_passbox=allow_passbox)


def dvr_creds(allow_passbox: bool | None = None) -> tuple[str, str]:
    return (secret("DVR_USER", field="dvr-user", allow_passbox=allow_passbox),
            secret("DVR_PASS", field="dvr-pass", allow_passbox=allow_passbox))


DVR_HOST = get("DVR_HOST", "dvr.local")
HA_URL = get("HA_URL", "http://homeassistant.local")
OLLAMA_URL = get("OLLAMA_URL", "http://127.0.0.1:11434")
RUN_DIR = Path(get("VLMSEC_RUN", "~/.vlmsec")).expanduser()


def main() -> None:
    print(f"  config file : {CONF_PATH} ({'found' if _FILE else 'not found, using defaults'})")
    for key, value in (("DVR_HOST", DVR_HOST), ("HA_URL", HA_URL), ("OLLAMA_URL", OLLAMA_URL),
                       ("VLMSEC_RUN", str(RUN_DIR))):
        print(f"  {key:11s} = {value}")
    # reports only whether a source exists, never the value
    print(f"  HA_TOKEN    : {'set' if os.environ.get('HA_TOKEN') or _FILE.get('HA_TOKEN') else 'unset'}")
    print(f"  passbox     : {'enabled' if os.environ.get('HN_USE_PASSBOX') == '1' else 'opt-in only'}")


if __name__ == "__main__":
    main()
