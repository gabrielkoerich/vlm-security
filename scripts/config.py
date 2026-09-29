#!/usr/bin/env python3
"""Every host, path and secret this repo needs, resolved at runtime.

Nothing about a specific network is in the repository, so it can be published without a
sanitising pass. Values resolve in this order:

  1. environment variable            (what `passbox exec --env ...` sets)
  2. untracked local file            ~/.home_network.conf, HOME_NETWORK_CONF overrides
  3. passbox, ONLY when asked        opt in with HN_USE_PASSBOX=1 or secret(..., allow_passbox=True)
  4. a generic default, never a real address

**The daemons must never reach step 3.** frame_ring, vehicle_bridge and plate_collector run
under launchd 24/7, and passbox needs a real login session to raise Touch ID, so a call from a
launchd agent or over SSH hangs rather than fails. Those processes read the untracked file.
Use passbox for interactive ops scripts, preferably as one injected grant:

    passbox exec --env DVR_USER=security/dvr-user --env DVR_PASS=security/dvr-password -- \\
        python3 dvr_zones.py allred 2 3

A passbox entry holds many fields (first line is the secret, then `key: value`), so one box can
carry a whole network:  passbox get security/home-network --field dvr-host

Config file format, one KEY=value per line, # for comments:

    DVR_HOST=10.0.0.10
    VISION_HOST=10.0.0.20
    HA_URL=http://10.0.0.30:8123

  python3 config.py            # show what resolved; never fetches a secret
  python3 config.py selfcheck
"""
import os
import sys

CONF_PATH = os.environ.get("HOME_NETWORK_CONF", os.path.expanduser("~/.home_network.conf"))
PASSBOX_BOX = os.environ.get("HN_PASSBOX_BOX", "security/home-network")


def _load(path=None):
    vals = {}
    try:
        with open(path or CONF_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip()
    except OSError:
        pass
    return vals


_FILE = _load()


def get(key, default=""):
    """Non-secret config. Never touches passbox, so this is always safe to call."""
    return os.environ.get(key) or _FILE.get(key) or default


def require(key):
    """For values with no safe default. Fail loudly rather than pretend."""
    v = get(key)
    if not v:
        raise RuntimeError(
            f"{key} is not set. Put it in {CONF_PATH} or export it. "
            "This repo deliberately ships no real addresses or secrets.")
    return v


def passbox_field(field, box=None, timeout=120):
    """One field from a passbox entry. Interactive use only, this can raise Touch ID."""
    import subprocess
    try:
        r = subprocess.run(["passbox", "get", box or PASSBOX_BOX, "--field", field],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:
        return ""


def secret(env_key, creds_line=None, field=None, allow_passbox=None):
    """A credential: env, then the untracked creds file, then passbox only if allowed.

    allow_passbox defaults to HN_USE_PASSBOX=1, so a daemon that never sets it can never hang
    on a Touch ID prompt it has no session to answer.
    """
    v = os.environ.get(env_key) or _FILE.get(env_key)
    if v:
        return v
    if creds_line is not None:
        try:
            lines = open(CREDS_FILE).read().splitlines()
            if len(lines) > creds_line and lines[creds_line].strip():
                return lines[creds_line].strip()
        except OSError:
            pass
    if allow_passbox is None:
        allow_passbox = os.environ.get("HN_USE_PASSBOX") == "1"
    if allow_passbox and field:
        return passbox_field(field)
    return ""


# Hosts. These defaults are documentation, not anyone's network.
DVR_HOST = get("DVR_HOST", "dvr.local")
VISION_HOST = get("VISION_HOST", "vision.local")
HA_HOST = get("HA_HOST", "homeassistant.local")
HA_URL = get("HA_URL", f"http://{HA_HOST}")
OLLAMA_URL = get("OLLAMA_URL", "http://127.0.0.1:11434")
DOORWAY_HOST = get("DOORWAY_HOST", "doorway.local")
VISION_USER = get("VISION_USER", "vision")
VISION_HOME = get("VISION_HOME", f"/Users/{VISION_USER}")
CREDS_FILE = get("CREDS_FILE", os.path.expanduser("~/.vehicle_bridge_creds"))


def ha_token(allow_passbox=None):
    return secret("HA_TOKEN", creds_line=0, field="ha-token", allow_passbox=allow_passbox)


def dvr_creds(allow_passbox=None):
    return (secret("DVR_USER", creds_line=1, field="dvr-user", allow_passbox=allow_passbox),
            secret("DVR_PASS", creds_line=2, field="dvr-pass", allow_passbox=allow_passbox))


def selfcheck():
    """Defaults must not be real addresses, and a daemon must never reach passbox."""
    import re
    ipish = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    for name in ("dvr.local", "vision.local", "homeassistant.local"):
        assert not ipish.match(name), f"{name} default must not be an IP"
    saved = dict(os.environ)
    try:
        for k in ("NOPE_KEY", "HN_USE_PASSBOX", "HA_TOKEN"):
            os.environ.pop(k, None)
        try:
            require("NOPE_KEY")
            raise AssertionError("require() must raise on a missing key")
        except RuntimeError:
            pass
        os.environ["DVR_HOST"] = "10.9.9.9"
        assert get("DVR_HOST") == "10.9.9.9", "env must win over file and default"
        # the important one: without opt-in, no passbox call can happen
        calls = []
        globals()["passbox_field"] = lambda *a, **k: calls.append(a) or "x"
        secret("DEFINITELY_UNSET_KEY", creds_line=None, field="ha-token")
        assert not calls, "secret() must not call passbox unless allowed"
        secret("DEFINITELY_UNSET_KEY", creds_line=None, field="ha-token", allow_passbox=True)
        assert calls, "secret() must call passbox when explicitly allowed"
    finally:
        os.environ.clear()
        os.environ.update(saved)
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        selfcheck()
    else:
        print(f"  config file: {CONF_PATH} ({'found' if _FILE else 'NOT found, using defaults'})")
        for k in ("DVR_HOST", "VISION_HOST", "HA_HOST", "HA_URL", "OLLAMA_URL", "VISION_HOME"):
            print(f"  {k:12s} = {globals()[k]}")
        # deliberately does not fetch: only reports whether a source exists
        have_file = os.path.exists(CREDS_FILE)
        print(f"  creds file   : {CREDS_FILE} ({'present' if have_file else 'absent'})")
        print(f"  HA_TOKEN env : {'set' if os.environ.get('HA_TOKEN') else 'unset'}")
        print(f"  passbox      : {'enabled' if os.environ.get('HN_USE_PASSBOX') == '1' else 'opt-in only'}")
