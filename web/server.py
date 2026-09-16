"""FastAPI backend for the browser demo (deployable on Render's free tier).

Responsibilities
----------------
* Serve the static frontend (``web/static``) - the frontend runs the model
  *in the browser* with onnxruntime-web, so the server does no work for the
  webcam stream.
* Serve the ONNX model at ``/model/model.onnx`` for the browser to download.
* ``POST /api/detect`` - server-side inference with ONNX Runtime (CPU) as a
  fallback for browsers without WebAssembly/WebGPU and for API clients.
* ``GET /api/health`` / ``GET /api/info`` - liveness + model metadata.

The model file is resolved in this order:
1. ``MODEL_PATH`` env var
2. ``web/models/model.onnx`` (commit it, or let ``scripts/prepare_web_model.py``
   put it there)
3. downloaded from ``MODEL_URL`` env var into ``web/models/model.onnx`` at
   startup (e.g. a GitHub Release asset).

No torch, no ultralytics, no OpenCV - the whole backend fits in ~250 MB RAM.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("web")

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
MODELS_DIR = WEB_DIR / "models"
CONFIG_PATH = ROOT / "configs" / "default.yaml"

# --------------------------------------------------------------------------- #
# Configuration (class names / colours / thresholds come from the same YAML the
# training code uses, so the web app can never drift from the model).
# --------------------------------------------------------------------------- #


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_cfg()
CLASS_NAMES: List[str] = CFG["class_names"]
CLASS_COLORS_BGR: Dict[str, List[int]] = CFG["class_colors"]
DEFAULT_CONF = float(os.environ.get("CONF_THRESHOLD", CFG["confidence_threshold"]))
DEFAULT_IOU = float(os.environ.get("IOU_THRESHOLD", CFG["iou_threshold"]))
MAX_DET = int(CFG.get("max_detections", 300))


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def resolve_model_path() -> Path:
    """Find (or download) the ONNX model. Raises with an actionable message."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    local = MODELS_DIR / "model.onnx"
    if local.exists():
        return local
    url = os.environ.get("MODEL_URL")
    if url:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".part")
        log.info("Downloading model from %s ...", url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "masked-face-detection/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(local)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download MODEL_URL={url}: {exc}") from exc
        log.info("Model saved to %s (%.1f MB)", local, local.stat().st_size / 1e6)
        return local
    raise RuntimeError(
        "No ONNX model found. Either commit web/models/model.onnx (run "
        "scripts/prepare_web_model.py), set MODEL_PATH, or set MODEL_URL to a "
        "downloadable .onnx (e.g. a GitHub Release asset)."
    )


class OnnxDetector:
    """Minimal YOLOv8 ONNX runner: letterbox -> session.run -> decode -> NMS."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [1, 3, H, W]
        self.input_h = int(shape[2]) if isinstance(shape[2], int) else int(CFG["image_size"])
        self.input_w = int(shape[3]) if isinstance(shape[3], int) else int(CFG["image_size"])
        self.output_name = self.session.get_outputs()[0].name
        self.model_path = model_path
        log.info("ONNX model loaded: %s (input %dx%d)", model_path.name, self.input_w, self.input_h)

    # ---- pre / post -------------------------------------------------------- #

    def letterbox(self, img: Image.Image) -> Tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio and pad with grey to the model size."""
        w0, h0 = img.size
        r = min(self.input_w / w0, self.input_h / h0)
        nw, nh = max(1, int(round(w0 * r))), max(1, int(round(h0 * r)))
        resized = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.input_w, self.input_h), (114, 114, 114))
        dx, dy = (self.input_w - nw) // 2, (self.input_h - nh) // 2
        canvas.paste(resized, (dx, dy))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None], r, dx, dy

    def detect(self, img: Image.Image, conf: float, iou: float) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        blob, r, dx, dy = self.letterbox(img)
        out = self.session.run([self.output_name], {self.input_name: blob})[0]  # (1, 4+nc, N)
        preds = out[0].T  # (N, 4+nc)
        boxes_xywh, scores = preds[:, :4], preds[:, 4:]
        cls = scores.argmax(1)
        confs = scores[np.arange(len(scores)), cls]
        keep = confs >= conf
        boxes_xywh, cls, confs = boxes_xywh[keep], cls[keep], confs[keep]
        # xywh (letterboxed px) -> xyxy (original px)
        x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - dx) / r
        y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - dy) / r
        x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - dx) / r
        y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - dy) / r
        w0, h0 = img.size
        xyxy = np.stack([x1.clip(0, w0), y1.clip(0, h0), x2.clip(0, w0), y2.clip(0, h0)], 1)
        idx = nms_per_class(xyxy, confs, cls, iou)[:MAX_DET]
        dets = [
            {
                "class_id": int(cls[i]),
                "class_name": CLASS_NAMES[int(cls[i])] if int(cls[i]) < len(CLASS_NAMES) else str(int(cls[i])),
                "confidence": round(float(confs[i]), 4),
                "bbox_xyxy": [round(float(v), 1) for v in xyxy[i]],
            }
            for i in idx
        ]
        dets.sort(key=lambda d: -d["confidence"])
        return dets, (time.perf_counter() - t0) * 1000


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thresh: float) -> List[int]:
    """Greedy non-maximum suppression, applied independently per class."""
    keep: List[int] = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        order = idx[np.argsort(-scores[idx])]
        while len(order):
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
            order = rest[iou < iou_thresh]
    return sorted(keep, key=lambda i: -scores[i])


def summarise(dets: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {n: 0 for n in CLASS_NAMES}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    counts["total"] = len(dets)
    return counts


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title="Masked Face Detection", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_detector: Optional[OnnxDetector] = None
_model_error: Optional[str] = None


@app.on_event("startup")
def _load_model() -> None:
    global _detector, _model_error
    try:
        _detector = OnnxDetector(resolve_model_path())
    except Exception as exc:  # noqa: BLE001 - keep serving the frontend, report via /api/info
        _model_error = str(exc)
        log.error("Model not available: %s", exc)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": _detector is not None}


@app.get("/api/info")
def info() -> Dict[str, Any]:
    """Everything the frontend needs to configure itself."""
    return {
        "class_names": CLASS_NAMES,
        "class_colors_rgb": {k: [v[2], v[1], v[0]] for k, v in CLASS_COLORS_BGR.items()},
        "box_label": CFG.get("box_label"),      # text drawn on boxes; null -> class name
        "confidence_threshold": DEFAULT_CONF,
        "iou_threshold": DEFAULT_IOU,
        "model_url": "/model/model.onnx",
        "model_loaded": _detector is not None,
        "model_input": [_detector.input_w, _detector.input_h] if _detector else None,
        "model_error": _model_error,
        "model_variant": CFG.get("model_variant"),
    }


@app.get("/model/model.onnx")
def model_file():
    """Serve the ONNX model so onnxruntime-web can run it in the browser."""
    if _detector is None:
        raise HTTPException(503, f"Model not available: {_model_error}")
    return FileResponse(str(_detector.model_path), media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(..., description="JPEG/PNG image"),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=0.99),
    iou: float = Query(DEFAULT_IOU, ge=0.1, le=0.95),
) -> JSONResponse:
    """Server-side detection. Returns detections in original-image pixels."""
    if _detector is None:
        raise HTTPException(503, f"Model not available: {_model_error}")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Not a decodable image: {exc}") from exc
    dets, ms = _detector.detect(img, conf, iou)
    return JSONResponse({
        "summary": summarise(dets),
        "detections": dets,
        "image_size": [img.width, img.height],
        "inference_ms": round(ms, 1),
    })


# Static frontend last so API routes take precedence.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":  # local dev: python web/server.py
    import uvicorn

    uvicorn.run("web.server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
