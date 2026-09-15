"""Shared helpers: configuration, logging, reproducibility, device, drawing.

Every other module imports from here. Nothing in this file depends on
Ultralytics so it can be used by the SSD fallback path as well.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def get_logger(name: str = "mfd", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger that prints to stdout (visible in Colab cells).

    Args:
        name: Logger name. Sub-modules pass ``__name__``.
        level: Logging level; defaults to INFO.

    Returns:
        A configured :class:`logging.Logger`. Handlers are added only once so
        repeated calls (e.g. re-running a Colab cell) do not duplicate output.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def load_config(
    path: "os.PathLike[str] | str | None" = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` and optionally apply overrides.

    Args:
        path: Path to a YAML config. Defaults to ``configs/default.yaml``.
        overrides: Flat ``{key: value}`` mapping applied on top of the file.
            Nested keys can be addressed with dots, e.g. ``"export.opset"``.

    Returns:
        The merged configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is malformed or missing required keys.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found at {cfg_path}. Did you clone the repo and run from its root?"
        )
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {cfg_path}: {exc}") from exc

    required = ["seed", "model_variant", "num_classes", "class_names", "image_size", "epochs"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {cfg_path} is missing required keys: {missing}")

    if len(cfg["class_names"]) != cfg["num_classes"]:
        raise ValueError(
            f"num_classes={cfg['num_classes']} but class_names has "
            f"{len(cfg['class_names'])} entries"
        )

    for key, value in (overrides or {}).items():
        _set_nested(cfg, key, value)
    return cfg


def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for ``dotted_key="a.b.c"``."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def save_config(cfg: Dict[str, Any], path: "os.PathLike[str] | str") -> Path:
    """Persist a config dict as YAML (used to snapshot the config next to a run).

    Args:
        cfg: Configuration dictionary.
        path: Destination ``.yaml`` path; parent dirs are created.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return out


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA) for reproducible runs.

    Args:
        seed: The seed value. The project default is 42.
        deterministic: If True, also switch cuDNN into deterministic mode.
            This costs a little speed but makes runs repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch not installed yet (e.g. running dataset prep only)
        log.warning("PyTorch not available; only Python/NumPy seeds were set.")
    log.info("Seeds set to %d (deterministic=%s)", seed, deterministic)


# --------------------------------------------------------------------------- #
# Device / memory
# --------------------------------------------------------------------------- #


@dataclass
class DeviceInfo:
    """Summary of the compute device the run will use."""

    device: str  # "0" for first CUDA GPU, "cpu" otherwise
    name: str
    total_memory_gb: float
    is_gpu: bool


def get_device(prefer_gpu: bool = True) -> DeviceInfo:
    """Detect the best available device and warn loudly if there is no GPU.

    Args:
        prefer_gpu: If False, force CPU even when CUDA is available.

    Returns:
        :class:`DeviceInfo` describing the selected device. ``device`` is in
        the string form Ultralytics expects (``"0"`` or ``"cpu"``).
    """
    try:
        import torch

        if prefer_gpu and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info = DeviceInfo("0", props.name, props.total_memory / 1e9, True)
            log.info("Using GPU: %s (%.1f GB)", info.name, info.total_memory_gb)
            return info
    except ImportError:
        pass
    log.warning(
        "No GPU detected. Training will be VERY slow. In Colab go to "
        "Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4)."
    )
    return DeviceInfo("cpu", "cpu", 0.0, False)


def free_memory() -> None:
    """Release cached CUDA memory and run the garbage collector.

    Call between major steps (train -> evaluate -> export) so a 12 GB Colab
    runtime does not OOM.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> Optional[Path]:
    """Mount Google Drive if in Colab; return the mount path or None.

    Args:
        mount_point: Where Drive is mounted inside the Colab VM.

    Returns:
        The mount path when Drive is available, otherwise ``None`` (local run).

    Raises:
        RuntimeError: If the mount fails (with an actionable message).
    """
    if not in_colab():
        log.info("Not running in Colab; skipping Drive mount.")
        return None
    try:
        from google.colab import drive

        drive.mount(mount_point, force_remount=False)
        log.info("Google Drive mounted at %s", mount_point)
        return Path(mount_point)
    except Exception as exc:  # pragma: no cover - Colab only
        raise RuntimeError(
            "Failed to mount Google Drive. Re-run the cell and accept the "
            "authorisation popup, or check your Google account permissions."
        ) from exc


# --------------------------------------------------------------------------- #
# Drawing helpers (shared by inference, evaluation, and the Gradio demo)
# --------------------------------------------------------------------------- #


def class_color(cfg: Dict[str, Any], class_name: str) -> Tuple[int, int, int]:
    """Look up the BGR colour for a class name from the config.

    Args:
        cfg: Loaded configuration.
        class_name: One of ``cfg["class_names"]``.

    Returns:
        A ``(B, G, R)`` tuple. Unknown classes fall back to white.
    """
    color = cfg.get("class_colors", {}).get(class_name, [255, 255, 255])
    return int(color[0]), int(color[1]), int(color[2])


def summarise_counts(class_ids: Sequence[int], cfg: Dict[str, Any]) -> Dict[str, int]:
    """Count detections per class plus a ``total`` entry.

    Args:
        class_ids: Predicted class index per detection.
        cfg: Loaded configuration.

    Returns:
        Dict keyed by every class name and ``"total"``.
    """
    names: List[str] = cfg["class_names"]
    counts = {n: 0 for n in names}
    for cid in class_ids:
        if 0 <= int(cid) < len(names):
            counts[names[int(cid)]] += 1
    counts["total"] = int(sum(counts[n] for n in names))
    return counts


def format_summary(counts: Dict[str, int]) -> str:
    """Render the corner banner text, e.g. ``"Faces: 5 | Masked: 3 | Unmasked: 2"``."""
    return (
        f"Faces: {counts.get('total', 0)} | Masked: {counts.get('with_mask', 0)} | "
        f"Unmasked: {counts.get('without_mask', 0)} | "
        f"Incorrect: {counts.get('mask_worn_incorrectly', 0)}"
    )


def draw_detections(
    image_bgr: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    cfg: Dict[str, Any],
    show_summary: bool = True,
) -> np.ndarray:
    """Draw colour-coded boxes, ``"label 0.94"`` text and a summary banner.

    Args:
        image_bgr: Input image (H, W, 3) in BGR. It is copied, not modified.
        boxes_xyxy: Boxes as ``[x1, y1, x2, y2]`` in absolute pixels.
        class_ids: Integer class index per box.
        confidences: Confidence score per box.
        cfg: Loaded configuration (for class names and colours).
        show_summary: Draw the ``"Faces: N | Masked: a | Unmasked: b"`` banner.

    Returns:
        The annotated image (BGR).
    """
    out = image_bgr.copy()
    names: List[str] = cfg["class_names"]
    h, w = out.shape[:2]
    thickness = max(1, int(round(min(h, w) / 320)))
    font_scale = max(0.4, min(h, w) / 1000)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), cid, conf in zip(boxes_xyxy, class_ids, confidences):
        name = names[int(cid)] if 0 <= int(cid) < len(names) else str(cid)
        color = class_color(cfg, name)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        ty = max(p1[1] - th - baseline - 2, 0)
        cv2.rectangle(out, (p1[0], ty), (p1[0] + tw + 4, ty + th + baseline + 4), color, -1)
        cv2.putText(out, label, (p1[0] + 2, ty + th + 2), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    if show_summary:
        banner = format_summary(summarise_counts(class_ids, cfg))
        (tw, th), baseline = cv2.getTextSize(banner, font, font_scale, thickness)
        cv2.rectangle(out, (0, 0), (tw + 12, th + baseline + 12), (0, 0, 0), -1)
        cv2.putText(out, banner, (6, th + 6), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (OpenCV) image to RGB (matplotlib / Gradio)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_dir(path: "os.PathLike[str] | str") -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Run / checkpoint locations
# --------------------------------------------------------------------------- #


def resolve_save_dir(cfg: Dict[str, Any]) -> Path:
    """Return the run directory, falling back to a local path outside Colab.

    ``cfg["save_dir"]`` points at Google Drive. When Drive is not mounted
    (local development or a Colab session without Drive) we fall back to
    ``<repo>/runs`` so the code still works and nothing silently fails.

    Args:
        cfg: Loaded configuration.

    Returns:
        An existing directory to write runs into.
    """
    save_dir = Path(cfg["save_dir"])
    wants_drive = str(save_dir).replace("\\", "/").startswith("/content/drive")
    if wants_drive and not Path("/content/drive/MyDrive").exists():
        fallback = PROJECT_ROOT / "runs"
        log.warning("Google Drive not mounted; saving runs to %s instead.", fallback)
        return ensure_dir(fallback)
    return ensure_dir(save_dir)


def run_dir(cfg: Dict[str, Any]) -> Path:
    """Return ``<save_dir>/<run_name>`` (Ultralytics' project/name layout)."""
    return resolve_save_dir(cfg) / cfg["run_name"]


def find_last_checkpoint(cfg: Dict[str, Any]) -> Optional[Path]:
    """Locate ``<save_dir>/<run_name>/weights/last.pt`` if it exists.

    Used by the auto-resume logic in notebook 02 and :mod:`src.train`.
    """
    ckpt = run_dir(cfg) / "weights" / "last.pt"
    return ckpt if ckpt.exists() else None


def find_best_checkpoint(cfg: Dict[str, Any]) -> Optional[Path]:
    """Locate ``<save_dir>/<run_name>/weights/best.pt`` if it exists."""
    ckpt = run_dir(cfg) / "weights" / "best.pt"
    return ckpt if ckpt.exists() else None
