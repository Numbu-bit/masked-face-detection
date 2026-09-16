"""Evaluation: metrics, plots, failure analysis and speed benchmarks.

Everything here works on a trained ``ultralytics.YOLO`` model plus the
prepared dataset from notebook 01. Figures are returned as matplotlib
``Figure`` objects (so notebooks can display them) *and* saved to
``<run_dir>/eval/`` on Drive.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.dataset import IMAGE_EXTS, read_yolo_label_file, yolo_to_xyxy
from src.utils import bgr_to_rgb, draw_detections, ensure_dir, free_memory, get_logger, legend_text, run_dir

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Metrics from Ultralytics validation
# --------------------------------------------------------------------------- #


@dataclass
class EvalReport:
    """Container for the headline numbers produced by :func:`evaluate_yolo`."""

    map50: float
    map50_95: float
    precision: float
    recall: float
    per_class: Dict[str, Dict[str, float]]  # name -> {ap50, ap50_95, precision, recall, f1}
    confusion_matrix: np.ndarray            # (nc+1, nc+1) incl. background
    pr_curves: Optional[Tuple[np.ndarray, np.ndarray]]  # (recall_x[1000], precision[nc,1000])
    save_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable version (used to write ``metrics.json``)."""
        return {
            "mAP50": self.map50, "mAP50_95": self.map50_95,
            "precision": self.precision, "recall": self.recall,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def evaluate_yolo(model: Any, cfg: Dict[str, Any], data_yaml: str, split: str = "test") -> EvalReport:
    """Run Ultralytics validation on a split and collect all metrics.

    Args:
        model: Trained ``ultralytics.YOLO``.
        cfg: Loaded configuration.
        data_yaml: Path to ``data.yaml``.
        split: ``"val"`` or ``"test"``.

    Returns:
        :class:`EvalReport`.
    """
    names: List[str] = cfg["class_names"]
    out_dir = ensure_dir(run_dir(cfg) / "eval")
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(cfg["image_size"]),
        batch=int(cfg["batch_size"]),
        conf=0.001,                # standard for mAP computation
        iou=float(cfg["iou_threshold"]),
        plots=True,
        project=str(out_dir),
        name=split,
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    per_class: Dict[str, Dict[str, float]] = {}
    idx_map = {int(c): i for i, c in enumerate(box.ap_class_index)}
    for cid, name in enumerate(names):
        if cid in idx_map:
            i = idx_map[cid]
            p, r = float(box.p[i]), float(box.r[i])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class[name] = {
                "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
                "precision": p, "recall": r, "f1": f1,
            }
        else:  # class absent from this split
            per_class[name] = {"ap50": 0.0, "ap50_95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pr_curves = None
    try:
        curves = box.curves_results  # [[px, py(nc,1000), 'Recall', 'Precision'], ...]
        if curves:
            px, py = curves[0][0], curves[0][1]
            pr_curves = (np.asarray(px), np.asarray(py))
    except Exception:  # older/newer ultralytics without curves_results
        pass

    report = EvalReport(
        map50=float(box.map50), map50_95=float(box.map),
        precision=float(box.mp), recall=float(box.mr),
        per_class=per_class,
        confusion_matrix=np.asarray(metrics.confusion_matrix.matrix),
        pr_curves=pr_curves,
        save_dir=out_dir,
    )
    with open(out_dir / f"metrics_{split}.json", "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    free_memory()
    return report


def print_report(report: EvalReport, cfg: Dict[str, Any]) -> None:
    """Pretty-print the headline metrics and a per-class table."""
    print(f"\n{'=' * 64}\n  mAP@0.5      : {report.map50:.4f}\n  mAP@0.5:0.95 : {report.map50_95:.4f}")
    print(f"  Precision    : {report.precision:.4f}\n  Recall       : {report.recall:.4f}\n{'=' * 64}")
    print(f"{'class':<24}{'AP50':>8}{'AP50-95':>10}{'P':>8}{'R':>8}{'F1':>8}")
    for name, m in report.per_class.items():
        print(f"{name:<24}{m['ap50']:>8.3f}{m['ap50_95']:>10.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(report: EvalReport, cfg: Dict[str, Any], normalize: bool = True):
    """Heat-map of the (nc+1)x(nc+1) confusion matrix, background included.

    Ultralytics stores rows = predicted, columns = true.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(cfg["class_names"]) + ["background"]
    cm = report.confusion_matrix.astype(float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion matrix" + (" (normalised by true class)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(report.save_dir / "confusion_matrix.png", dpi=150)
    return fig


def plot_pr_curves(report: EvalReport, cfg: Dict[str, Any]):
    """Per-class precision-recall curves (falls back to Ultralytics' PNG)."""
    import matplotlib.pyplot as plt

    if report.pr_curves is None:
        png = next(report.save_dir.rglob("PR_curve.png"), None)
        if png is None:
            log.warning("No PR-curve data available.")
            return None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(bgr_to_rgb(cv2.imread(str(png))))
        ax.axis("off")
        return fig

    px, py = report.pr_curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for cid, name in enumerate(cfg["class_names"]):
        if cid < py.shape[0]:
            ap = report.per_class[name]["ap50"]
            ax.plot(px, py[cid], label=f"{name} (AP50={ap:.3f})", linewidth=2)
    ax.plot(px, py.mean(0), "k--", label=f"all classes (mAP50={report.map50:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Precision-Recall (IoU=0.5)")
    fig.tight_layout()
    fig.savefig(report.save_dir / "pr_curves.png", dpi=150)
    return fig


def plot_training_curves(cfg: Dict[str, Any]):
    """Loss / mAP / LR over epochs from Ultralytics' ``results.csv``."""
    import matplotlib.pyplot as plt
    import pandas as pd

    csv = run_dir(cfg) / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found - has training (notebook 02) completed at least one epoch?")
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss"):
        if key in df:
            axes[0].plot(df["epoch"], df[key], label=key)
    axes[0].set_title("Losses"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
        if key in df:
            axes[1].plot(df["epoch"], df[key], label=key.replace("(B)", ""))
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    for key in ("lr/pg0", "lr/pg1", "lr/pg2"):
        if key in df:
            axes[2].plot(df["epoch"], df[key], label=key)
    axes[2].set_title("Learning rate"); axes[2].set_xlabel("epoch"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ensure_dir(run_dir(cfg) / "eval") / "training_curves.png", dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# Qualitative results
# --------------------------------------------------------------------------- #


def list_split_images(data_root: Path, split: str) -> List[Path]:
    """Sorted image paths for a split."""
    d = Path(data_root) / split / "images"
    if not d.exists():
        raise FileNotFoundError(f"{d} missing - run notebook 01 first.")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def predict_boxes(model: Any, image_bgr: np.ndarray, cfg: Dict[str, Any], conf: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run YOLO on one BGR image and return ``(xyxy, class_ids, confidences)``."""
    res = model.predict(
        image_bgr,
        imgsz=int(cfg["image_size"]),
        conf=float(cfg["confidence_threshold"] if conf is None else conf),
        iou=float(cfg["iou_threshold"]),
        max_det=int(cfg.get("max_detections", 300)),
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))
    return (
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy().astype(int),
        res.boxes.conf.cpu().numpy(),
    )


def plot_prediction_grid(model: Any, cfg: Dict[str, Any], data_root: Path, split: str = "test", n: int = 16, seed: int = 42):
    """Grid of ``n`` random images with predicted, colour-coded boxes."""
    import matplotlib.pyplot as plt

    paths = list_split_images(data_root, split)
    rng = random.Random(seed)
    chosen = rng.sample(paths, min(n, len(paths)))
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    for ax, path in zip(np.ravel(axes), chosen):
        img = cv2.imread(str(path))
        boxes, ids, confs = predict_boxes(model, img, cfg)
        ax.imshow(bgr_to_rgb(draw_detections(img, boxes, ids, confs, cfg, show_summary=False)))
        ax.set_title(path.stem[:24], fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(chosen):]:
        ax.axis("off")
    fig.suptitle(f"Predictions on {len(chosen)} random {split} images  ({legend_text(cfg)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(ensure_dir(run_dir(cfg) / "eval") / f"prediction_grid_{split}.png", dpi=120)
    return fig


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N,4)`` / ``(M,4)`` xyxy arrays -> ``(N,M)``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass
class DetectionCase:
    """One detection with its correctness, used for failure analysis."""

    image_path: Path
    box: List[float]
    pred_class: int
    conf: float
    true_class: Optional[int]  # None when the box matched no GT (false positive)
    correct: bool


def collect_detection_cases(model: Any, cfg: Dict[str, Any], data_root: Path, split: str = "test",
                            iou_thresh: float = 0.5, conf: float = 0.25, max_images: Optional[int] = None) -> List[DetectionCase]:
    """Match every prediction to ground truth and label it correct / wrong.

    A prediction is *correct* when it overlaps a not-yet-matched GT box of
    the same class with IoU >= ``iou_thresh``. Class confusions and boxes on
    background are *wrong*.
    """
    from tqdm.auto import tqdm

    names_n = len(cfg["class_names"])
    cases: List[DetectionCase] = []
    paths = list_split_images(data_root, split)
    if max_images:
        paths = paths[:max_images]
    for path in tqdm(paths, desc="analysing detections", leave=False):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_rows = read_yolo_label_file(Path(data_root) / split / "labels" / f"{path.stem}.txt")
        gt_boxes, gt_ids = yolo_to_xyxy(gt_rows, w, h)
        gt_boxes_np, gt_ids_np = np.asarray(gt_boxes).reshape(-1, 4), np.asarray(gt_ids, dtype=int)
        boxes, ids, confs = predict_boxes(model, img, cfg, conf=conf)
        order = np.argsort(-confs)
        ious = box_iou(boxes, gt_boxes_np)
        matched = np.zeros(len(gt_boxes_np), dtype=bool)
        for i in order:
            true_cls: Optional[int] = None
            correct = False
            if ious.shape[1]:
                cand = np.where((ious[i] >= iou_thresh) & ~matched)[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
                    true_cls = int(gt_ids_np[j])
                    correct = true_cls == int(ids[i]) and 0 <= true_cls < names_n
                    if correct:
                        matched[j] = True
            cases.append(DetectionCase(path, boxes[i].tolist(), int(ids[i]), float(confs[i]), true_cls, correct))
    return cases


def plot_failure_cases(cases: Sequence[DetectionCase], cfg: Dict[str, Any], k: int = 10):
    """Show the k lowest-confidence *correct* and k highest-confidence *wrong* detections."""
    import matplotlib.pyplot as plt

    names = cfg["class_names"]
    correct = sorted((c for c in cases if c.correct), key=lambda c: c.conf)[:k]
    wrong = sorted((c for c in cases if not c.correct), key=lambda c: -c.conf)[:k]

    def crop(case: DetectionCase) -> np.ndarray:
        img = cv2.imread(str(case.image_path))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = case.box
        pad = 0.25 * max(x2 - x1, y2 - y1)
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        vis = draw_detections(img, [case.box], [case.pred_class], [case.conf], cfg, show_summary=False)
        return bgr_to_rgb(vis[y1:y2, x1:x2])

    fig, axes = plt.subplots(2, k, figsize=(2.4 * k, 5.6))
    for row, (group, title) in enumerate(((correct, "lowest-confidence CORRECT"), (wrong, "highest-confidence WRONG"))):
        for col in range(k):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(group):
                c = group[col]
                ax.imshow(crop(c))
                gt = "bg" if c.true_class is None else names[c.true_class]
                ax.set_title(f"pred {names[c.pred_class]}\ngt {gt}  conf {c.conf:.2f}", fontsize=7)
        axes[row, 0].text(-0.1, 0.5, title, transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ensure_dir(run_dir(cfg) / "eval") / "failure_cases.png", dpi=120)
    return fig


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #


def benchmark_fps(model: Any, cfg: Dict[str, Any], data_root: Path, n_images: int = 50, warmup: int = 5) -> Dict[str, float]:
    """Measure end-to-end inference FPS on GPU (if available) and CPU.

    Args:
        model: Trained ``ultralytics.YOLO``.
        cfg: Loaded configuration.
        data_root: Prepared dataset root (uses the test split).
        n_images: Number of timed images per device.
        warmup: Untimed warm-up iterations.

    Returns:
        ``{"gpu_fps": x, "cpu_fps": y}`` (``gpu_fps`` is ``0.0`` without CUDA).
    """
    import torch

    paths = list_split_images(data_root, "test")[: n_images + warmup]
    images = [cv2.imread(str(p)) for p in paths]
    images = [im for im in images if im is not None]
    results: Dict[str, float] = {"gpu_fps": 0.0, "cpu_fps": 0.0}

    for device_key, device in (("gpu_fps", 0), ("cpu_fps", "cpu")):
        if device == 0 and not torch.cuda.is_available():
            continue
        for im in images[:warmup]:
            model.predict(im, imgsz=int(cfg["image_size"]), device=device, verbose=False)
        if device == 0:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed = images[warmup:]
        for im in timed:
            model.predict(im, imgsz=int(cfg["image_size"]), device=device, verbose=False)
        if device == 0:
            torch.cuda.synchronize()
        results[device_key] = len(timed) / max(time.perf_counter() - t0, 1e-9)
        log.info("%s: %.1f FPS over %d images", device_key, results[device_key], len(timed))
    if torch.cuda.is_available():  # leave the model on GPU for subsequent cells
        model.predict(images[0], imgsz=int(cfg["image_size"]), device=0, verbose=False)
    return results


# --------------------------------------------------------------------------- #
# Target check + suggestions
# --------------------------------------------------------------------------- #


def check_targets(report: EvalReport, fps: Dict[str, float], cfg: Dict[str, Any]) -> List[str]:
    """Compare results with ``cfg["targets"]`` and return actionable suggestions.

    An empty list means every target was met.
    """
    targets = cfg["targets"]
    tips: List[str] = []
    if report.map50 < targets["map50"]:
        tips.append(
            f"mAP@0.5 = {report.map50:.3f} < {targets['map50']}. Try: (1) train longer "
            f"(epochs > {cfg['epochs']}), (2) switch model_variant to a larger model "
            "(yolov8s -> yolov8m), (3) add more data (Roboflow/MAFA), (4) raise image_size to 640/800."
        )
    weak = [n for n, m in report.per_class.items() if m["f1"] < targets["per_class_f1"]]
    if weak:
        tips.append(
            f"Per-class F1 below {targets['per_class_f1']} for {weak}. These are usually the "
            "rare classes: oversample them, add synthetic mask overlays, or lower mixup/mosaic "
            "so small faces are not destroyed by augmentation."
        )
    if fps.get("gpu_fps", 0.0) and fps["gpu_fps"] < targets["fps_gpu"]:
        tips.append(
            f"GPU inference = {fps['gpu_fps']:.1f} FPS < {targets['fps_gpu']}. Use yolov8n, "
            "reduce image_size to 416/480, or export to TensorRT/ONNX with FP16."
        )
    return tips
