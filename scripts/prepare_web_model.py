"""Export ``best.pt`` to the ONNX file the web app (web/) serves and runs in-browser.

Why a separate export from ``export_model.py``?
* The browser runs on CPU/WebGPU, so a smaller input (default 416 px) gives a
  usable live-webcam frame rate; the training size (640) is available via ``--imgsz``.
* The file is written to a fixed location (``web/models/model.onnx``) that the
  FastAPI server and the Render deployment expect.
* Static shapes, opset 12, no NMS in the graph - what onnxruntime-web handles best.

Usage (repo root)::

    python scripts/prepare_web_model.py                      # best.pt from the run dir, 416 px
    python scripts/prepare_web_model.py --weights best.pt --imgsz 640
    python scripts/prepare_web_model.py --check photo.jpg    # also run the ONNX on an image

Then either commit ``web/models/model.onnx`` (~43 MB for yolov8s) or upload it
somewhere and set ``MODEL_URL`` on Render.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import build_yolo  # noqa: E402
from src.utils import find_best_checkpoint, get_logger, load_config  # noqa: E402

log = get_logger("web-export")
DEST = ROOT / "web" / "models" / "model.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best.pt for the browser demo")
    parser.add_argument("--weights", default=None, help="Checkpoint (.pt); default: run dir best.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--imgsz", type=int, default=416, help="Input size (416 = fast in-browser; 640 = training size)")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--check", default=None, help="Optional image to run through the exported ONNX")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights) if args.weights else find_best_checkpoint(cfg)
    if weights is None or not weights.exists():
        raise FileNotFoundError("No best.pt found. Pass --weights or train first (notebook 02).")

    model = build_yolo(cfg, weights=str(weights))
    log.info("Exporting %s -> ONNX (imgsz=%d, opset=%d, simplify=True)", weights.name, args.imgsz, args.opset)
    out = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False, half=False))

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, dest)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)

    # Sanity-check the graph with onnxruntime (same library the server uses).
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    log.info("input  %s %s", inp.name, inp.shape)
    log.info("output %s %s  (expect [1, %d, N])", outp.name, outp.shape, 4 + cfg["num_classes"])
    assert list(inp.shape) == [1, 3, args.imgsz, args.imgsz], f"unexpected input shape {inp.shape}"
    assert outp.shape[1] == 4 + cfg["num_classes"], f"unexpected output channels {outp.shape}"

    if args.check:
        sys.path.insert(0, str(ROOT / "web"))
        from PIL import Image

        from server import OnnxDetector  # type: ignore

        det = OnnxDetector(dest)
        dets, ms = det.detect(Image.open(args.check), cfg["confidence_threshold"], cfg["iou_threshold"])
        log.info("%s: %d detections in %.0f ms -> %s", args.check, len(dets), ms,
                 [(d["class_name"], d["confidence"]) for d in dets[:5]])
    else:
        dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        sess.run(None, {inp.name: dummy})
        log.info("onnxruntime forward pass OK")
    shown = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
    print(f"\nDone. Next: commit {shown} (or upload it and set MODEL_URL), then deploy.")


if __name__ == "__main__":
    main()
