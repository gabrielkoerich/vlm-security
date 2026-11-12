"""A rolling buffer of live frames per camera, so an event can read the seconds before it.

One ffmpeg per camera reads the live RTSP stream and writes JPEGs at RING_FPS into
RING_ROOT/<channel>/. A pruner drops anything older than RING_SECONDS.

Why it exists: a motion event arrives seconds after the subject, and reading those seconds back
from DVR playback is slow, and many DVRs refuse a window that recent. Frames captured live are
already on disk, so the bridge slices around the event at once.

Lessons built in:

  * Cap the frame size. A high resolution camera can be several times the pixels of the others
    for no gain in detection. The cap is square, so a portrait camera is not squeezed sideways.
  * A running ffmpeg is not a working one. It can stay alive and write nothing, so a watchdog
    kills a channel that has produced no frame for WATCHDOG_STALL_S.
  * On macOS, Local Network privacy is granted per binary. A Homebrew ffmpeg started by launchd
    can be refused the LAN while python is allowed, so python relays the RTSP connection on
    loopback (RTSP_PROXY=1) and ffmpeg only ever talks to 127.0.0.1.
  * Channels overlooking interior space are dropped after all other configuration, so no setting
    can route them into the ring.

    GET /health   per channel frame count and age of the newest frame
    GET /frame?ch=N   the newest frame, or 503 when the ring is empty

    python -m vlmsec.frame_ring
"""

import http.server
import json
import os
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import override
from urllib.parse import parse_qs, urlparse

from vlmsec import config

FFMPEG = config.get("FFMPEG", "ffmpeg")
ROOT = Path(config.get("RING_ROOT", "/tmp/framering"))
PORT = config.get_int("RING_PORT", 8099)
RTSP_PORT = 554
PROXY = config.get("RTSP_PROXY", "0") == "1"
PROXY_PORT = 5540
RING_FPS = 2
RING_SECONDS = 45
WATCHDOG_TICK_S = 20
WATCHDOG_STALL_S = 60
MAX_SIDE = 1280
# Every DVR names its streams differently, {ch} is replaced by the channel number
RTSP_PATH = config.get("RTSP_PATH", "/ch{ch}")


def channels(raw: str, indoor_raw: str) -> tuple[int, ...]:
    """Outdoor channels only. The indoor list is applied last, it is a floor, not a preference."""
    indoor = {int(c) for c in indoor_raw.split(",") if c.strip()}
    return tuple(int(c) for c in raw.split(",") if c.strip() and int(c) not in indoor)


CHANNELS = channels(config.get("RING_CHANNELS", "1,2,3,4"), config.get("RING_INDOOR_CHANNELS", ""))


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def frames_in_window(ch: int, window: float, end: float | None = None, root: Path = ROOT) -> list[str]:
    """Ring files whose mtime falls in [end - window, end], oldest first."""
    end = time.time() if end is None else end
    out: list[tuple[float, str]] = []
    try:
        entries = list((root / str(ch)).iterdir())
    except FileNotFoundError:
        return []
    for p in entries:
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if end - window <= m <= end + 0.5:
            out.append((m, str(p)))
    out.sort()
    return [p for _, p in out]


def relay(conn: socket.socket) -> None:
    """A dumb byte relay from loopback to the DVR. With RTSP over TCP everything is one connection."""
    try:
        up = socket.create_connection((config.DVR_HOST, RTSP_PORT), timeout=10)
    except OSError as e:
        log(f"proxy: cannot reach the DVR: {type(e).__name__}")
        conn.close()
        return
    done = threading.Event()

    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while not done.is_set():
                chunk = src.recv(65536)
                if not chunk:
                    break
                dst.sendall(chunk)
        except OSError:
            pass
        finally:
            done.set()

    threading.Thread(target=pump, args=(up, conn), daemon=True).start()
    pump(conn, up)
    up.close()
    conn.close()


def proxy_loop() -> None:
    srv = socket.create_server(("127.0.0.1", PROXY_PORT))
    while True:
        conn = srv.accept()[0]
        threading.Thread(target=relay, args=(conn,), daemon=True).start()


def capture_loop(ch: int) -> None:
    """One ffmpeg per camera, restarted when it exits. Never log the URL, it holds the password."""
    d = ROOT / str(ch)
    d.mkdir(parents=True, exist_ok=True)
    user, pw = config.dvr_creds(allow_passbox=False)
    host = f"127.0.0.1:{PROXY_PORT}" if PROXY else config.DVR_HOST
    url = f"rtsp://{user}:{pw}@{host}{RTSP_PATH.format(ch=ch)}"
    scale = f"scale=w={MAX_SIDE}:h={MAX_SIDE}:force_original_aspect_ratio=decrease"
    while True:
        done = subprocess.run(
            [FFMPEG, "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", url,
             "-vf", f"fps={RING_FPS},{scale}", "-q:v", "4", "-f", "image2", str(d / "f_%06d.jpg")],
            capture_output=True, check=False)
        # ffmpeg echoes the input URL on error, so the password is scrubbed before logging
        err = done.stderr.decode("utf-8", "replace").replace(pw, "***") if pw else ""
        log(f"ch{ch} capture ended rc={done.returncode}: {' '.join(err.split())[:200]}")
        time.sleep(5)


def watchdog_loop() -> None:
    last_seen = dict.fromkeys(CHANNELS, time.time())
    while True:
        time.sleep(WATCHDOG_TICK_S)
        for ch in CHANNELS:
            if frames_in_window(ch, WATCHDOG_STALL_S):
                last_seen[ch] = time.time()
            elif time.time() - last_seen[ch] >= WATCHDOG_STALL_S:
                log(f"ch{ch} produced nothing for {WATCHDOG_STALL_S}s, killing its ffmpeg")
                subprocess.run(["pkill", "-f", RTSP_PATH.format(ch=ch)], capture_output=True, check=False)
                last_seen[ch] = time.time()


def prune_loop() -> None:
    while True:
        cutoff = time.time() - RING_SECONDS
        for ch in CHANNELS:
            for p in (ROOT / str(ch)).glob("*.jpg"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass
        time.sleep(2)


class Handler(http.server.BaseHTTPRequestHandler):
    @override
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/health":
            body: dict[str, dict[str, float | int | None]] = {}
            for ch in CHANNELS:
                fs = frames_in_window(ch, RING_SECONDS)
                newest = os.path.getmtime(fs[-1]) if fs else None
                body[str(ch)] = {"frames": len(fs),
                                 "age_s": round(time.time() - newest, 1) if newest else None}
            self._send(json.dumps(body).encode(), "application/json")
        elif u.path == "/frame":
            ch = int(q.get("ch", ["0"])[0])
            got = frames_in_window(ch, 10) if ch in CHANNELS else []
            if not got:
                # never serve an old frame here, a caller that caches the last good image would
                # keep showing it as live
                self.send_error(503, f"ring empty for ch{ch}")
                return
            self._send(Path(got[-1]).read_bytes(), "image/jpeg")
        else:
            self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads: bool = True
    allow_reuse_address: bool = True


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROXY:
        threading.Thread(target=proxy_loop, daemon=True).start()
    for ch in CHANNELS:
        threading.Thread(target=capture_loop, args=(ch,), daemon=True).start()
    threading.Thread(target=prune_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    log(f"frame ring on 127.0.0.1:{PORT}, channels {CHANNELS} at {RING_FPS} fps, {RING_SECONDS}s deep")
    Server(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
