#!/usr/bin/env python3
"""Rolling live-frame ring per camera, with motion-based frame picking.

Runs on the vision host. One ffmpeg per camera reads the live RTSP stream and writes JPEGs at RING_FPS
into /tmp/framering/<ch>/. A pruner drops anything older than RING_SECONDS.

Why this exists: an event needs the seconds BEFORE it, and reading those from DVR playback was
both slow and, from launchd, impossible (see RTSPProxy, ffmpeg is blocked from the LAN). Frames
captured live are already on disk, so an event can slice around it instantly.

Why motion picking: the subject is visible for a very short time. The 21:15 motorbike was in
frame for 1.5s out of a 20s window, and a fixed offset misses it. Scoring frames by how much
they differ from their neighbour finds the frames that actually contain the moving subject.

Serves over HTTP so Home Assistant can use it as a generic camera:
  GET /frame?ch=4&window=10        -> single JPEG, the peak-motion frame in the last 10s
  GET /health                      -> json per-camera frame counts and age

Hosts and creds come from config (env, then the untracked ~/.home_network.conf
and ~/.vehicle_bridge_creds). Nothing about this network is in the repo.
Deployed via launchd (com.vlmsec.frame-ring). No numpy, no PIL, ffmpeg only.
"""
import http.server, json, os, shutil, socket, socketserver, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DVR = config.DVR_HOST
FFMPEG = config.get("FFMPEG", "/opt/homebrew/bin/ffmpeg")
ROOT = "/tmp/framering"
BURST_ROOT = "/tmp/framering_bursts"
BURST_KEEP = 40            # keep only recent bursts, they are the biggest thing on disk
# The stream advertises 100fps in its container but only averages 15, so an uncapped capture
# wrote 998 duplicate frames and 153MB for 10 seconds. Cap at the real rate.
BURST_FPS = 15
PORT = 8099
PROXY_PORT = 5540        # loopback RTSP relay, see RTSPProxy
RING_FPS = 2
RING_SECONDS = 45          # deep enough that the async collector can still reach back
WATCHDOG_TICK_S = 20       # how often to check each channel is still producing
WATCHDOG_STALL_S = 60      # no frames for this long means the stream stalled, restart it
THUMB_W, THUMB_H = 64, 36  # motion scoring resolution, tiny on purpose
# Square cap, not 1280x720: a portrait fisheye such as 2304x2592 hits a landscape box sideways
# and gets squeezed to 640 wide, throwing away the detail that made it worth pointing there
MAX_W, MAX_H = 1280, 1280
# Outdoor channels only. A DVR app's "CAM N" usually maps to RTSP channel N
CHANNELS = tuple(int(c) for c in os.environ.get("RING_CHANNELS", "1,2,3,4").split(","))
# Channels overlooking interior space must never enter the ring or reach the model
# This is a floor, not a preference, so it is applied after any other configuration
INDOOR = {int(c) for c in os.environ.get("RING_INDOOR_CHANNELS", "").split(",") if c.strip()}
CHANNELS = tuple(c for c in CHANNELS if c not in INDOOR)

def creds():
    """Read lazily: the selfcheck and the pickers must not need DVR credentials.

    Never passbox here: this runs under launchd with no session to answer Touch ID.
    """
    user, pw = config.dvr_creds(allow_passbox=False)
    if not (user and pw):
        raise RuntimeError(f"no DVR credentials, see {config.CREDS_FILE}")
    return user, pw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class RTSPProxy(socketserver.ThreadingTCPServer):
    """Loopback to DVR TCP relay.

    macOS Local Network privacy gates network access per binary. python3 is approved, the
    Homebrew ffmpeg is not, so ffmpeg spawned from launchd gets EHOSTUNREACH ("No route to
    host") talking to the DVR while python3 on the same agent connects fine. Verified
    2026-09-26 with a launchd probe. Loopback is not gated, so python3 holds the real socket
    and ffmpeg only ever connects to 127.0.0.1. With -rtsp_transport tcp everything is
    interleaved on this one connection, so a dumb byte relay is enough.
    """
    daemon_threads = True
    allow_reuse_address = True


class RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            up = socket.create_connection((DVR, 554), timeout=10)
        except Exception as e:
            log(f"proxy: cannot reach DVR: {type(e).__name__}")
            return
        done = threading.Event()

        def pump(src, dst):
            try:
                while not done.is_set():
                    b = src.recv(65536)
                    if not b:
                        break
                    dst.sendall(b)
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=pump, args=(up, self.request), daemon=True)
        t.start()
        pump(self.request, up)
        done.set()
        for s in (up, self.request):
            try:
                s.close()
            except Exception:
                pass


def capture_loop(ch):
    """One ffmpeg per camera, restarted if it dies. Never log the url, it holds the password."""
    d = os.path.join(ROOT, str(ch))
    os.makedirs(d, exist_ok=True)
    user, pw = creds()
    # via the loopback proxy: ffmpeg is blocked from the LAN by Local Network privacy
    url = f"rtsp://{user}:{pw}@127.0.0.1:{PROXY_PORT}/cam/realmonitor?channel={ch}&subtype=0"
    while True:
        err, rc = "", "?"
        try:
            r = subprocess.run(
                [FFMPEG, "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
                 "-i", url,
                 # cap the frame: a fisheye can be several times the resolution of the other cameras
                 "-vf", f"fps={RING_FPS},scale=w={MAX_W}:h={MAX_H}:force_original_aspect_ratio=decrease",
                 "-q:v", "4", "-f", "image2", os.path.join(d, "f_%06d.jpg")],
                capture_output=True)
            rc, err = r.returncode, r.stderr.decode("utf-8", "replace")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        # ffmpeg echoes the input url on error, so scrub the password before it reaches the log
        err = err.replace(pw, "***").replace(f"{user}:***", "***")
        log(f"ch{ch} capture ended rc={rc}: {' '.join(err.split())[:200] or '(no stderr)'}")
        time.sleep(5)


def watchdog_loop():
    """Kill a channel's ffmpeg if it stops producing, even though it has not exited.

    The doorway stalled for over two hours on 2026-09-27: ffmpeg stayed alive, wrote no
    frames, logged nothing, and HA's generic camera kept serving its last cached image, so the
    AI analysed a two-hour-old frame. A process that is running is not a process that is
    working.
    """
    last_seen = {ch: time.time() for ch in CHANNELS}
    while True:
        time.sleep(WATCHDOG_TICK_S)
        for ch in CHANNELS:
            fresh = frames_in_window(ch, WATCHDOG_STALL_S)
            if fresh:
                last_seen[ch] = time.time()
                continue
            stalled = time.time() - last_seen[ch]
            if stalled < WATCHDOG_STALL_S:
                continue
            log(f"ch{ch} produced nothing for {int(stalled)}s, killing its ffmpeg")
            subprocess.run(["pkill", "-f", f"channel={ch}&subtype"], capture_output=True)
            last_seen[ch] = time.time()


def prune_loop():
    while True:
        cutoff = time.time() - RING_SECONDS
        for ch in CHANNELS:
            d = os.path.join(ROOT, str(ch))
            try:
                for n in os.listdir(d):
                    p = os.path.join(d, n)
                    try:
                        if os.path.getmtime(p) < cutoff:
                            os.remove(p)
                    except OSError:
                        pass
            except FileNotFoundError:
                pass
        time.sleep(2)


def burst(ch, seconds, fps=BURST_FPS):
    """Capture the next `seconds` from the live stream at native rate, into its own directory.

    The ring holds the seconds BEFORE an event at 2fps, which is enough to notice something but
    not enough to identify it: on 2026-09-27 a car was present in 36 of 433 native-rate frames
    and in none of the 3 the ring picked. This grabs the pass itself, forward from now, and
    does not touch DVR playback, which refuses recent windows and blocks rather than erroring.
    """
    if ch in INDOOR:
        raise ValueError(f"channel {ch} is indoor, refusing")
    d = os.path.join(BURST_ROOT, f"{ch}_{int(time.time())}")
    os.makedirs(d, exist_ok=True)
    user, pw = creds()
    url = f"rtsp://{user}:{pw}@127.0.0.1:{PROXY_PORT}/cam/realmonitor?channel={ch}&subtype=0"
    vf = f"fps={fps}," if fps else ""
    try:
        subprocess.run(
            [FFMPEG, "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
             "-i", url, "-t", str(seconds),
             "-vf", f"{vf}scale=w={MAX_W}:h={MAX_H}:force_original_aspect_ratio=decrease",
             "-q:v", "4", "-f", "image2", os.path.join(d, "b_%05d.jpg")],
            capture_output=True, timeout=seconds + 25)
    except subprocess.TimeoutExpired:
        log(f"burst ch{ch}: ffmpeg overran, keeping what it wrote")
    except Exception as e:
        log(f"burst ch{ch}: {type(e).__name__}")
    got = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
    log(f"burst ch{ch}: {len(got)} frames over {seconds}s -> {d}")
    return got


def prune_bursts():
    """Bursts are big. Keep only the most recent ones."""
    try:
        dirs = sorted(os.listdir(BURST_ROOT))
    except FileNotFoundError:
        return
    for name in dirs[:-BURST_KEEP]:
        shutil.rmtree(os.path.join(BURST_ROOT, name), ignore_errors=True)


def frames_in_window(ch, window, end=None):
    """Ring files whose mtime falls in [end-window, end], oldest first."""
    d = os.path.join(ROOT, str(ch))
    end = end or time.time()
    out = []
    try:
        for n in os.listdir(d):
            p = os.path.join(d, n)
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if end - window <= m <= end + 0.5:
                out.append((m, p))
    except FileNotFoundError:
        return []
    out.sort()
    return [p for _, p in out]


def thumbs(paths):
    """Decode every jpeg to one tiny grayscale frame in a SINGLE ffmpeg call."""
    if not paths:
        return []
    lst = "\n".join(f"file '{p}'" for p in paths)
    try:
        r = subprocess.run(
            [FFMPEG, "-nostdin", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-protocol_whitelist", "file,pipe", "-i", "-",
             "-vf", f"scale={THUMB_W}:{THUMB_H},format=gray", "-f", "rawvideo", "-"],
            input=lst.encode(), capture_output=True, timeout=20)
    except Exception:
        return []
    n = THUMB_W * THUMB_H
    raw = r.stdout
    return [raw[i * n:(i + 1) * n] for i in range(len(raw) // n)]


def motion_scores(tl):
    """Mean absolute difference against the previous thumbnail. First frame scores 0."""
    if not tl:
        return []
    scores = [0.0]
    for a, b in zip(tl, tl[1:]):
        if len(a) != len(b) or not a:
            scores.append(0.0)
            continue
        scores.append(sum(abs(x - y) for x, y in zip(a, b)) / len(a))
    return scores


def pick(ch, window=10, n=1, end=None, order="motion"):
    """The n frames with the most motion in the window.

    order="motion" returns the busiest frame first, which lets a consumer asking one frame at a
    time stop on the first call in the common case. order="time" keeps chronological order.
    """
    paths = frames_in_window(ch, window, end)
    if not paths:
        return []
    tl = thumbs(paths)
    if len(tl) != len(paths):
        return paths[-n:]                       # scoring failed, fall back to the newest
    scored = sorted(zip(motion_scores(tl), paths), key=lambda t: -t[0])[:n]
    if order == "motion":
        return [p for _, p in scored]
    chosen = {p for _, p in scored}
    return [p for p in paths if p in chosen]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/health":
            body = {}
            for ch in CHANNELS:
                fs = frames_in_window(ch, RING_SECONDS)
                newest = max((os.path.getmtime(p) for p in fs), default=0)
                body[str(ch)] = {"frames": len(fs),
                                 "age_s": round(time.time() - newest, 1) if newest else None}
            out = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        if u.path == "/burst":
            ch = int(q.get("ch", ["4"])[0])
            secs = min(float(q.get("seconds", ["10"])[0]), 60)
            fps = float(q.get("fps", [str(BURST_FPS)])[0])
            try:
                got = burst(ch, secs, fps)
            except ValueError as e:
                self.send_error(400, str(e))
                return
            prune_bursts()
            out = json.dumps({"ch": ch, "frames": got, "count": len(got)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        if u.path == "/frame":
            ch = int(q.get("ch", ["4"])[0])
            window = float(q.get("window", ["10"])[0])
            got = pick(ch, window, 1)
            if not got:
                # never serve something old here: HA's generic camera caches the last good
                # fetch and will happily show a two-hour-old frame if this ever succeeds late
                self.send_error(503, f"ring empty for ch{ch}")
                return
            data = open(got[0], "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def selfcheck():
    """Motion scoring must rank a changed frame above identical ones."""
    flat = bytes([100]) * 16
    moved = bytes([100] * 8 + [200] * 8)
    s = motion_scores([flat, flat, moved, moved])
    assert s[1] == 0.0, s
    assert s[2] > 0.0, s
    assert s.index(max(s)) == 2, s
    assert motion_scores([]) == []
    assert motion_scores([flat]) == [0.0]
    assert not (set(CHANNELS) & INDOOR), f"indoor channel in ring: {CHANNELS}"
    try:
        burst(7, 1)
        raise AssertionError("burst must refuse an indoor channel")
    except ValueError:
        pass
    print("selfcheck ok")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        selfcheck()
        return
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(BURST_ROOT, exist_ok=True)
    proxy = RTSPProxy(("127.0.0.1", PROXY_PORT), RelayHandler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    log(f"rtsp proxy on 127.0.0.1:{PROXY_PORT} -> {DVR}:554")
    for ch in CHANNELS:
        threading.Thread(target=capture_loop, args=(ch,), daemon=True).start()
    threading.Thread(target=prune_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    log(f"frame_ring up on :{PORT}, {len(CHANNELS)} cameras at {RING_FPS}fps, {RING_SECONDS}s ring")
    Server(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
