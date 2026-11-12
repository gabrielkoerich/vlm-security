"""Object detector as a small HTTP service on loopback, so the bridge never loads torch itself.

Loading the weights takes seconds, so the model stays in memory and a frame costs tens of ms.

    GET  /health
    POST /detect  {"paths": ["/tmp/a.jpg", ...], "conf": 0.25}
         -> {"results": [{"path": ..., "detections": [{"cls", "conf", "box": [cx, cy, w, h]}]}]}

It decides nothing on its own. The detector is good at "is there a person" and bad at "is it on
the property", which is the opposite of the VLM, so each does the half it is good at.

    uv sync --group vision && uv run python -m vlmsec.detector_service
"""

import http.server
import json
import socketserver
import threading
from pathlib import Path
from typing import TYPE_CHECKING, override

from vlmsec import config, net

if TYPE_CHECKING:
    from ultralytics import YOLO

MODEL_NAME = config.get("DETECTOR_MODEL", "yolo11s.pt")
PORT = config.get_int("DETECTOR_PORT", 8098)
IMGSZ = config.get_int("DETECTOR_IMGSZ", 960)
DEVICE = config.get("DETECTOR_DEVICE", "cpu")
BATCH = 16
# COCO ids worth reporting, person first because it drives the alarm
WANT = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog"}

_model: "YOLO | None" = None
_lock = threading.Lock()


def model() -> "YOLO":
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(MODEL_NAME)
    return _model


def release() -> None:
    """Give the GPU cache back after each request.

    PyTorch keeps every batch's GPU buffers for reuse, so a long running detector grows to
    gigabytes of GPU memory it does not need. On a machine that also runs the VLM, that memory
    is taken from the model.
    """
    import torch

    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()


def detect(paths: list[str], conf: float) -> list[dict[str, object]]:
    todo = [p for p in paths if Path(p).is_file()]
    out: list[dict[str, object]] = [{"path": p, "error": "missing"} for p in paths if p not in todo]
    m = model()
    for i in range(0, len(todo), BATCH):
        for r in m.predict(todo[i:i + BATCH], imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False):
            dets: list[dict[str, object]] = []
            if r.boxes is not None:
                for c, p, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywh.tolist(),
                                     strict=True):
                    if int(c) in WANT:
                        dets.append({"cls": WANT[int(c)], "conf": round(p, 3),
                                     "box": [round(v, 1) for v in box[:4]]})
            out.append({"path": r.path, "detections": dets})
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    @override
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, code: int, body: object) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self._send(200, {"model": MODEL_NAME, "device": DEVICE, "loaded": _model is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.startswith("/detect"):
            self._send(404, {"error": "not found"})
            return
        try:
            req = net.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        except ValueError as e:
            self._send(400, {"error": type(e).__name__})
            return
        raw = net.field(req, "paths")
        paths = [net.as_str(p) for p in raw] if net.is_list(raw) else []
        conf = net.as_float(net.field(req, "conf"), 0.25)
        # one request at a time: one GPU context, and concurrency buys nothing here
        with _lock:
            try:
                self._send(200, {"results": detect(paths, conf)})
            except (OSError, RuntimeError) as e:
                self._send(500, {"error": type(e).__name__})
            finally:
                release()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads: bool = True
    allow_reuse_address: bool = True


def main() -> None:
    model()
    print(f"detector on 127.0.0.1:{PORT} with {MODEL_NAME} on {DEVICE}", flush=True)
    Server(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
