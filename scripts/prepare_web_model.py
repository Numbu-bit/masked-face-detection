"""Export ``best.pt`` to the ONNX file the web app (web/) serves and runs in-browser.

Why a separate export from ``export_model.py``?
* The browser runs on CPU/WebGPU, so a smaller input (default 416 px) gives a
  usable live-webcam frame rate; the training size (640) is available via ``--imgsz``.
* The file is written to a fixed location (``web/static/models/model.onnx``) that
  the FastAPI server, a static-site deployment and the Render blueprint expect.
* ``web/static/config.json`` is regenerated from ``configs/default.yaml`` so a
  static deployment (no Python backend) has the class names / colours / thresholds.
* Static shapes, opset 12, no NMS in the graph - what onnxruntime-web handles best.

Usage (repo root)::

    python scripts/prepare_web_model.py                      # best.pt from the run dir -> 416 px + 320 px
    python scripts/prepare_web_model.py --weights best.pt --imgsz 640 416 320
    python scripts/prepare_web_model.py --check photo.jpg    # also run the ONNX on an image

Then either commit ``web/static/models/model.onnx`` (~43 MB for yolov8s) or upload
it somewhere and set ``MODEL_URL`` on Render (web-service deployment only).
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
DEST = ROOT / "web" / "static" / "models" / "model.onnx"


def write_static_config(cfg: dict, imgsz: int, models: dict | None = None,
                        path: Path = ROOT / "web" / "static" / "config.json") -> Path:
    """Write the subset of the config the browser needs when there is no API (static deploy)."""
    import json

    payload = {
        "class_names": cfg["class_names"],
        "class_colors_rgb": {k: [v[2], v[1], v[0]] for k, v in cfg["class_colors"].items()},
        "box_label": cfg.get("box_label"),
        "confidence_threshold": float(cfg["confidence_threshold"]),
        "iou_threshold": float(cfg["iou_threshold"]),
        "model_url": "models/model.onnx",
        "models": models or {str(imgsz): "models/model.onnx"},
        "model_input": [imgsz, imgsz],
        "model_variant": cfg.get("model_variant"),
        "model_loaded": True,
        "static": True,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best.pt for the browser demo")
    parser.add_argument("--weights", default=None, help="Checkpoint (.pt); default: run dir best.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[416, 320],
                        help="Input size(s). First -> model.onnx (default); others -> model_<N>.onnx. "
                             "416 = accurate, 320 = fast live webcam, 640 = training size")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--check", default=None, help="Optional image to run through the exported ONNX")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights) if args.weights else find_best_checkpoint(cfg)
    if weights is None or not weights.exists():
        raise FileNotFoundError("No best.pt found. Pass --weights or train first (notebook 02).")

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sizes = list(dict.fromkeys(int(v) for v in args.imgsz))  # de-duplicate, keep order
    models: dict = {}
    for i, imgsz in enumerate(sizes):
        model = build_yolo(cfg, weights=str(weights))  # fresh model per export (export mutates it)
        log.info("Exporting %s -> ONNX (imgsz=%d, opset=%d, simplify=True)", weights.name, imgsz, args.opset)
        out = Path(model.export(format="onnx", imgsz=imgsz, opset=args.opset, simplify=True, dynamic=False, half=False))
        target = dest if i == 0 else dest.with_name(f"{dest.stem}_{imgsz}{dest.suffix}")
        shutil.copy2(out, target)
        models[str(imgsz)] = f"models/{target.name}"
        log.info("Saved %s (%.1f MB)", target, target.stat().st_size / 1e6)

        # Sanity-check the graph with onnxruntime (same library the server uses).
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
        log.info("input  %s %s | output %s %s  (expect [1, %d, N])", inp.name, inp.shape, outp.name, outp.shape, 4 + cfg["num_classes"])
        assert list(inp.shape) == [1, 3, imgsz, imgsz], f"unexpected input shape {inp.shape}"
        assert outp.shape[1] == 4 + cfg["num_classes"], f"unexpected output channels {outp.shape}"
        sess.run(None, {inp.name: np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)})
        log.info("onnxruntime forward pass OK")

    write_static_config(cfg, sizes[0], models=models)

    if args.check:
        sys.path.insert(0, str(ROOT / "web"))
        from PIL import Image

        from server import OnnxDetector  # type: ignore

        det = OnnxDetector(dest)
        dets, ms = det.detect(Image.open(args.check), cfg["confidence_threshold"], cfg["iou_threshold"])
        log.info("%s: %d detections in %.0f ms -> %s", args.check, len(dets), ms,
                 [(d["class_name"], d["confidence"]) for d in dets[:5]])
    shown = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
    print(f"\nDone. Next: commit {shown} (or upload it and set MODEL_URL), then deploy.")


if __name__ == "__main__":
    main()
