# Web app — browser demo + API, deployable on Render (free tier)

```
web/
├── server.py            # FastAPI: static frontend, /model/model.onnx, POST /api/detect, /api/info, /api/health
├── requirements.txt     # onnxruntime + fastapi + Pillow + numpy  (no torch — fits in 512 MB)
├── static/
│   ├── index.html       # UI (Tailwind CSS)
│   ├── app.js           # webcam / upload, onnxruntime-web session, drawing
│   ├── yolo.js          # pure pre/post-processing (letterbox, decode, NMS) — unit-tested
│   └── tailwind.css     # compiled from src/input.css (committed; no Node needed at deploy time)
├── src/input.css        # Tailwind source + component classes
├── tailwind.config.js · package.json
├── models/model.onnx    # <- your exported model (not in git by default; see below)
└── tests/               # node --test: unit tests + numerical parity with server.py
```

## How it works

1. The page calls `GET /api/info` (class names, colours, thresholds from `configs/default.yaml`).
2. It downloads `/model/model.onnx` and creates an **onnxruntime-web** session — WebGPU when the
   browser has it, otherwise multithreaded WASM.
3. Webcam frames (or an uploaded image) are letterboxed on a canvas, run through the model **in the
   browser**, decoded + NMS'd in `yolo.js`, and drawn with the same colours / labels / banner as the
   Python pipeline. Nothing is uploaded.
4. "On the server (API)" mode POSTs JPEG frames to `/api/detect` instead (onnxruntime CPU) — the
   automatic fallback for browsers without WASM, and a plain HTTP API for other clients:

```bash
curl -F file=@photo.jpg "https://<your-app>.onrender.com/api/detect?conf=0.45"
# {"summary": {"with_mask": 2, "without_mask": 1, "mask_worn_incorrectly": 0, "total": 3},
#  "detections": [{"class_id": 0, "class_name": "with_mask", "confidence": 0.93, "bbox_xyxy": [...]}, ...],
#  "image_size": [1280, 720], "inference_ms": 180.4}
```

## 1. Produce the model file

From notebook 04 (last cell) — or locally with the training environment:

```bash
python scripts/prepare_web_model.py --weights /path/to/best.pt          # 416 px (default, live-webcam friendly)
python scripts/prepare_web_model.py --weights /path/to/best.pt --imgsz 640
```

This writes `web/models/model.onnx` (~43 MB for yolov8s, ~12 MB for yolov8n).

## 2. Run locally

```bash
pip install -r web/requirements.txt
uvicorn web.server:app --reload --port 8000        # http://localhost:8000
```

Camera access needs a secure context: `localhost` is fine; any other host must be HTTPS
(Render gives you HTTPS automatically).

## 3. Deploy to Render

**Get the model to Render** — pick one:

| Option | How |
|---|---|
| **A. Commit it** (simplest) | `git add -f web/models/model.onnx && git commit -m "Add web model" && git push`. 43 MB is fine for GitHub (limit 100 MB). |
| **B. GitHub Release** | GitHub → *Releases → Draft a new release* → attach `model.onnx` → publish. Copy the asset URL and set it as `MODEL_URL` in Render (the server downloads it at startup). |

**Create the service:**

1. https://dashboard.render.com → **New → Blueprint** → connect `Numbu-bit/masked-face-detection`.
   Render reads `render.yaml` (Python 3.11, `pip install -r web/requirements.txt`,
   `uvicorn web.server:app`, health check `/api/health`, free plan).
2. If you chose option B, fill in `MODEL_URL` when prompted (or later under *Environment*).
3. Click **Apply**. First build takes ~3 min. Your app is at `https://masked-face-detection-xxxx.onrender.com`.

Every push to `main` redeploys automatically.

**Free-tier notes:** the service sleeps after 15 min without traffic; the first request afterwards
takes 30–60 s (the page shows "Downloading model…" while it wakes). Browser-mode detection is
unaffected by server load since the model runs on the visitor's device.

## Tests

```bash
cd web && npm install && npm test                       # unit tests for yolo.js
python web/tests/make_fixtures.py /tmp/fx               # (training env) then:
FIXTURES=/tmp/fx npm test                               # + parity: JS results == server.py results
npm run build:css                                       # after editing index.html / app.js / input.css
```
