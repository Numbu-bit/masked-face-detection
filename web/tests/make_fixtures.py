"""Generate parity-test fixtures for web/tests/yolo.test.mjs.

Usage (repo root, training environment)::

    python web/tests/make_fixtures.py <out_dir> [--image path.jpg]

Exports COCO-pretrained yolov8n to ONNX (416 px), runs the server-side
detector on one image and writes: coco.onnx, letterboxed.rgba, python_result.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--image", default=None, help="Test image (default: ultralytics bus.jpg sample)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils.downloads import safe_download

    import server  # web/server.py

    onnx = Path(YOLO("yolov8n.pt").export(format="onnx", imgsz=416, opset=12, simplify=True, dynamic=False, half=False))
    (out / "coco.onnx").write_bytes(onnx.read_bytes())

    if args.image is None:
        safe_download("https://ultralytics.com/images/bus.jpg", dir=out)
        args.image = str(out / "bus.jpg")
    img = Image.open(args.image)
    img.load()

    det = server.OnnxDetector(out / "coco.onnx")
    blob, r, dx, dy = det.letterbox(img)
    rgb = (blob[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], 2)
    (out / "letterboxed.rgba").write_bytes(rgba.tobytes())
    dets, ms = det.detect(img, 0.25, 0.5)
    json.dump({"image": args.image, "orig": [img.width, img.height], "ratio": r, "dx": dx, "dy": dy,
               "W": det.input_w, "H": det.input_h, "dets": dets}, open(out / "python_result.json", "w"), indent=1)
    print(f"{len(dets)} detections in {ms:.0f} ms -> fixtures in {out}")


if __name__ == "__main__":
    main()
