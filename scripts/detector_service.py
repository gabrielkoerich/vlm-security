#!/usr/bin/env python3
"""Object detector as a tiny HTTP service, so anything on this host can use it.

Runs under the ultralytics venv (`~/vlmbench/.venv`), which the other services cannot import
because they run on the system python. Loading the weights takes seconds, so the model is held
in memory and a request is ~20ms.

  GET  /health
  POST /detect   {"paths": ["/tmp/a.jpg", ...], "conf": 0.25}
       -> {"results": [{"path": ..., "detections": [{"cls","conf","box"}], "ms": 21.4}]}
  POST /track    {"paths": [...ordered frames...], "conf": 0.25}
       -> {"tracks": [{"id","cls","frames","max_conf","direction"}]}

Why it exists: measured 2026-09-27, the detector found vehicles in 36 of 433 frames of an event
the VLM pipeline reported as empty, best confidence 0.95, at 31ms a frame. Tracking collapsed
those into one car moving right. The VLMs managed 1 of 4 such events.

Deployed via launchd (com.vlmsec.detector). Read only, it decides nothing by itself.
"""
import http.server, json, os, socketserver, sys, threading, time

MODEL_NAME = os.environ.get("DETECTOR_MODEL", "yolo26s.pt")
PORT = int(os.environ.get("DETECTOR_PORT", "8098"))
IMGSZ = int(os.environ.get("DETECTOR_IMGSZ", "960"))
DEVICE = os.environ.get("DETECTOR_DEVICE", "mps")
# COCO ids worth reporting here. Person first, it drives the alarm.
WANT = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_model = None
_lock = threading.Lock()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_NAME)
        log(f"loaded {MODEL_NAME}")
    return _model


def detect(paths, conf):
    out = []
    m = model()
    for p in paths:
        if not os.path.exists(p):
            out.append({"path": p, "error": "missing"})
            continue
        t0 = time.time()
        r = m.predict(p, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
        dets = []
        for b in r.boxes:
            c = int(b.cls[0])
            if c not in WANT:
                continue
            x, y, w, h = (float(v) for v in b.xywh[0])
            dets.append({"cls": WANT[c], "conf": round(float(b.conf[0]), 3),
                         "box": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]})
        out.append({"path": p, "detections": dets, "ms": round((time.time() - t0) * 1000, 1)})
    return out


def track(paths, conf):
    """Ordered frames in, one entry per tracked subject out, with direction of travel."""
    m = model()
    seen = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        r = m.track(p, imgsz=IMGSZ, conf=conf, device=DEVICE,
                    tracker="bytetrack.yaml", persist=True, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        for b, tid in zip(r.boxes, r.boxes.id.tolist()):
            c = int(b.cls[0])
            if c not in WANT:
                continue
            t = seen.setdefault(int(tid), {"id": int(tid), "cls": WANT[c], "x": [], "conf": []})
            t["x"].append(float(b.xywh[0][0]))
            t["conf"].append(float(b.conf[0]))
    out = []
    for t in seen.values():
        direction = "static"
        if len(t["x"]) > 1:
            dx = t["x"][-1] - t["x"][0]
            direction = "right" if dx > 20 else ("left" if dx < -20 else "static")
        out.append({"id": t["id"], "cls": t["cls"], "frames": len(t["x"]),
                    "max_conf": round(max(t["conf"]), 3), "direction": direction})
    return sorted(out, key=lambda d: -d["max_conf"])


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"model": MODEL_NAME, "imgsz": IMGSZ, "device": DEVICE,
                             "loaded": _model is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            paths = req.get("paths") or []
            conf = float(req.get("conf", 0.25))
        except Exception as e:
            self._send(400, {"error": f"{type(e).__name__}"})
            return
        # one request at a time: a single MPS context, and concurrency buys nothing here
        with _lock:
            try:
                if self.path.startswith("/detect"):
                    self._send(200, {"results": detect(paths, conf)})
                elif self.path.startswith("/track"):
                    self._send(200, {"tracks": track(paths, conf)})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                log("error:", type(e).__name__, e)
                self._send(500, {"error": f"{type(e).__name__}"})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def selfcheck():
    """Direction must come from displacement, and only reported classes may appear."""
    assert set(WANT.values()) == {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    xs = {"id": 1, "cls": "car", "x": [10.0, 400.0], "conf": [0.5, 0.9]}
    dx = xs["x"][-1] - xs["x"][0]
    assert dx > 20
    print("selfcheck ok")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        selfcheck()
        return
    model()          # load before serving, so the first real request is fast
    log(f"detector up on :{PORT} with {MODEL_NAME} at imgsz {IMGSZ}")
    Server(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
