"""Export the best checkpoint to ONNX / TFLite / TorchScript and validate.

Each exported model is re-loaded through Ultralytics, run on N test images,
and its boxes are matched against the PyTorch original. The script fails
loudly if any matched pair has IoU below ``cfg["export"]["min_iou"]``.

Usage (from the repo root)::

    python scripts/export_model.py                       # uses best.pt from the run dir
    python scripts/export_model.py --weights path/to/best.pt --formats onnx torchscript
    python scripts/export_model.py --data-root /content/data --no-validate

TFLite export pulls in TensorFlow (~500 MB). It is attempted last and a
failure there does not abort the other formats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import box_iou, list_split_images  # noqa: E402
from src.model import build_yolo  # noqa: E402
from src.utils import find_best_checkpoint, free_memory, get_logger, load_config, set_seed  # noqa: E402

log = get_logger("export")

FORMAT_SUFFIX = {"onnx": ".onnx", "torchscript": ".torchscript", "tflite": "_saved_model"}


def export_one(model: Any, fmt: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Export ``model`` to one format; return the output path or None on failure."""
    exp = cfg["export"]
    kwargs: Dict[str, Any] = dict(format=fmt, imgsz=int(cfg["image_size"]))
    if fmt == "onnx":
        kwargs.update(opset=int(exp["opset"]), simplify=bool(exp["simplify"]))
    try:
        out = model.export(**kwargs)
        path = Path(out) if out else None
        log.info("%s -> %s", fmt, path)
        return path
    except Exception as exc:  # noqa: BLE001 - we want to keep going with other formats
        log.error("Export to %s FAILED: %s", fmt, exc)
        if fmt == "tflite":
            log.error("TFLite needs TensorFlow; run `pip install tensorflow onnx2tf` and retry.")
        return None


def _tflite_file(saved_model_dir: Path) -> Optional[Path]:
    """Locate the float32 .tflite file Ultralytics writes inside *_saved_model/."""
    if saved_model_dir.suffix == ".tflite":
        return saved_model_dir
    candidates = sorted(saved_model_dir.glob("*float32.tflite")) or sorted(saved_model_dir.glob("*.tflite"))
    return candidates[0] if candidates else None


def validate_export(
    reference: Any, exported_path: Path, cfg: Dict[str, Any], images: List[np.ndarray]
) -> Dict[str, Any]:
    """Compare exported-model boxes to the PyTorch reference on sample images.

    Returns:
        ``{"passed": bool, "min_iou": float, "mean_iou": float, "count_mismatch": int}``.
    """
    from ultralytics import YOLO

    exported = YOLO(str(exported_path), task="detect")
    # Floor the confidence so boxes hovering at a very low threshold cannot flip
    # in/out between runtimes and cause spurious count mismatches.
    kw = dict(imgsz=int(cfg["image_size"]), conf=max(float(cfg["confidence_threshold"]), 0.25),
              iou=float(cfg["iou_threshold"]), verbose=False)
    ious: List[float] = []
    count_mismatch = 0
    for img in images:
        ref = reference.predict(img, **kw)[0].boxes
        exp = exported.predict(img, **kw)[0].boxes
        ref_xyxy = ref.xyxy.cpu().numpy() if ref is not None else np.zeros((0, 4))
        exp_xyxy = exp.xyxy.cpu().numpy() if exp is not None else np.zeros((0, 4))
        if len(ref_xyxy) != len(exp_xyxy):
            count_mismatch += 1
        if len(ref_xyxy) and len(exp_xyxy):
            m = box_iou(ref_xyxy, exp_xyxy)
            ious.extend(m.max(axis=1).tolist())  # best match for every reference box
    min_iou = float(min(ious)) if ious else 1.0
    mean_iou = float(np.mean(ious)) if ious else 1.0
    passed = min_iou >= float(cfg["export"]["min_iou"]) and count_mismatch == 0
    return {"passed": passed, "min_iou": round(min_iou, 4), "mean_iou": round(mean_iou, 4),
            "count_mismatch": count_mismatch, "boxes_compared": len(ious)}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export best.pt to deployment formats")
    parser.add_argument("--weights", default=None, help="Checkpoint (.pt); default: run dir best.pt")
    parser.add_argument("--config", default=None, help="Config YAML")
    parser.add_argument("--formats", nargs="+", default=None, help="Subset of: onnx tflite torchscript")
    parser.add_argument("--data-root", default=None, help="Prepared dataset root (for validation images)")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    weights = Path(args.weights) if args.weights else find_best_checkpoint(cfg)
    if weights is None or not weights.exists():
        raise FileNotFoundError("No best.pt found. Train first or pass --weights.")
    formats: List[str] = args.formats or list(cfg["export"]["formats"])
    # TFLite last so a TensorFlow install problem cannot block the others.
    formats = sorted(formats, key=lambda f: f == "tflite")

    images: List[np.ndarray] = []
    if not args.no_validate:
        data_root = Path(args.data_root or cfg["data_root"])
        try:
            paths = list_split_images(data_root, "test")[: int(cfg["export"]["validation_images"])]
            images = [im for im in (cv2.imread(str(p)) for p in paths) if im is not None]
        except FileNotFoundError as exc:
            log.warning("%s - skipping validation", exc)
    if not images and not args.no_validate:
        log.warning("No validation images found; exports will not be validated.")

    results: Dict[str, Any] = {}
    for fmt in formats:
        model = build_yolo(cfg, weights=str(weights))  # fresh model per export (export mutates it)
        out_path = export_one(model, fmt, cfg)
        entry: Dict[str, Any] = {"path": str(out_path) if out_path else None}
        if out_path and images:
            check_path = _tflite_file(out_path) if fmt == "tflite" else out_path
            if check_path is None:
                entry["validation"] = {"passed": False, "error": "no .tflite file produced"}
            else:
                try:
                    reference = build_yolo(cfg, weights=str(weights))
                    entry["validation"] = validate_export(reference, check_path, cfg, images)
                except Exception as exc:  # noqa: BLE001
                    entry["validation"] = {"passed": False, "error": str(exc)}
            log.info("%s validation: %s", fmt, entry["validation"])
        results[fmt] = entry
        free_memory()

    report_path = weights.parent / "export_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    failed = [f for f, r in results.items() if r["path"] is None or not r.get("validation", {"passed": True})["passed"]]
    if failed:
        log.error("Problems with formats: %s (see %s)", failed, report_path)
        sys.exit(1)
    log.info("All exports succeeded and validated. Report: %s", report_path)


if __name__ == "__main__":
    main()
