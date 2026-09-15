"""End-to-end smoke test on a tiny synthetic dataset (no downloads, CPU-friendly).

Exercises every module in ``src/``: VOC->YOLO conversion, stratified split,
Albumentations, YOLOv8 training (1 epoch, 160 px), auto-resume, evaluation
plots, image / video / JPEG inference, the Gradio callback and ONNX export.
Runs in ~3 minutes on a laptop CPU.

Usage (from the repo root)::

    python scripts/smoke_test.py            # default: <repo>/.smoke (deleted first)
    python scripts/smoke_test.py --keep     # leave the outputs for inspection
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.dataset import collect_voc_samples, stratified_split, write_data_yaml, write_split  # noqa: E402
from src.evaluate import (  # noqa: E402
    benchmark_fps, check_targets, collect_detection_cases, evaluate_yolo, plot_confusion_matrix,
    plot_failure_cases, plot_pr_curves, plot_prediction_grid, plot_training_curves, print_report,
)
from src.inference import Detector, detections_to_json  # noqa: E402
from src.model import build_yolo, build_ssd_mobilenetv2  # noqa: E402
from src.train import train_yolo  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

log = get_logger("smoke")

VOC_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]
COLORS = {"with_mask": (0, 200, 0), "without_mask": (0, 0, 220), "mask_weared_incorrect": (0, 140, 255)}


def make_synthetic_voc(raw: Path, n: int = 48, seed: int = 0) -> None:
    """Write ``n`` 320x240 images with coloured rectangles + VOC XML labels."""
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "annotations").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    for i in range(n):
        w, h = 320, 240
        img = np_rng.integers(60, 180, (h, w, 3), dtype=np.uint8)
        objs = ""
        for _ in range(rng.randint(1, 3)):
            x1, y1 = rng.randint(0, w - 80), rng.randint(0, h - 80)
            x2, y2 = x1 + rng.randint(40, 79), y1 + rng.randint(40, 79)
            name = rng.choice(VOC_NAMES)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[name], -1)
            objs += (f"<object><name>{name}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                     f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")
        cv2.imwrite(str(raw / "images" / f"img{i}.png"), img)
        (raw / "annotations" / f"img{i}.xml").write_text(
            f"<annotation><filename>img{i}.png</filename><size><width>{w}</width>"
            f"<height>{h}</height></size>{objs}</annotation>", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(ROOT / ".smoke"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    work = Path(args.workdir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    cfg = load_config(overrides={
        "model_variant": "yolov8n", "image_size": 160, "batch_size": 8, "num_workers": 0,
        "save_dir": str(work / "runs"), "run_name": "smoke", "data_root": str(work / "data"),
        "epochs": 1, "patience": 5, "save_period": 1, "confidence_threshold": 0.01,
    })
    set_seed(cfg["seed"])

    log.info("1/7 dataset")
    make_synthetic_voc(work / "raw")
    samples = collect_voc_samples(work / "raw", cfg)
    split = stratified_split(samples, cfg["split_ratios"], cfg["seed"])
    data_root = write_split(split, work / "data")
    data_yaml = write_data_yaml(data_root, cfg["class_names"])

    log.info("2/7 models")
    import torch

    ssd = build_ssd_mobilenetv2(cfg, pretrained_backbone=False).eval()
    with torch.no_grad():
        ssd([torch.rand(3, 160, 160)])
    build_yolo(cfg)

    log.info("3/7 train + auto-resume")
    best = train_yolo(cfg, str(data_yaml), resume=False)
    assert best.exists(), "best.pt missing after training"
    train_yolo(cfg, str(data_yaml))  # resume=None on a finished run must be a no-op

    log.info("4/7 evaluation")
    model = build_yolo(cfg, weights=str(best))
    report = evaluate_yolo(model, cfg, str(data_yaml), split="test")
    print_report(report, cfg)
    plot_confusion_matrix(report, cfg)
    plot_pr_curves(report, cfg)
    plot_training_curves(cfg)
    plot_prediction_grid(model, cfg, data_root, n=4)
    plot_failure_cases(collect_detection_cases(model, cfg, data_root, max_images=5), cfg, k=3)
    fps = benchmark_fps(model, cfg, data_root, n_images=3, warmup=1)
    check_targets(report, fps, cfg)

    log.info("5/7 inference")
    det = Detector(cfg)
    img = cv2.imread(str(next((data_root / "test" / "images").iterdir())))
    annotated, dets = det.detect_image(img)
    assert annotated.shape == img.shape
    detections_to_json(dets, cfg)
    vid = work / "in.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 5, (img.shape[1], img.shape[0]))
    for _ in range(6):
        writer.write(img)
    writer.release()
    summary = det.process_video(vid, work / "out.mp4", show_progress=False)
    assert summary["frames"] == 6
    det.detect_frame_jpeg(cv2.imencode(".jpg", img)[1].tobytes())

    log.info("6/7 gradio callback")
    try:
        from demo.gradio_app import build_demo
        from src.utils import bgr_to_rgb

        demo = build_demo(cfg)
        demo.fns[0].fn(bgr_to_rgb(img), 0.25)
    except ImportError:
        log.warning("gradio not installed - skipping UI check")

    log.info("7/7 export (onnx)")
    import subprocess

    from src.utils import save_config

    snap = save_config(cfg, work / "cfg.yaml")
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_model.py"), "--config", str(snap),
                         "--weights", str(best), "--formats", "onnx"], capture_output=True,
                        encoding="utf-8", errors="replace")
    if rc.returncode != 0:
        print(rc.stdout[-2000:], rc.stderr[-2000:])
        raise SystemExit("export failed")

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
