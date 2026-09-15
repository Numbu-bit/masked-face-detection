"""Model definitions.

Primary : YOLOv8 (n/s/m) via the ``ultralytics`` package, fine-tuned from
          COCO weights. Everything (training, validation, export) is handled
          by Ultralytics; this module just standardises how we build/load it.
Fallback: SSD with a MobileNetV2 backbone assembled from ``torchvision``
          building blocks. Used only if Ultralytics is unavailable or broken
          in the Colab environment.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import find_best_checkpoint, get_logger

log = get_logger(__name__)

SUPPORTED_VARIANTS = ("yolov8n", "yolov8s", "yolov8m")


# --------------------------------------------------------------------------- #
# YOLOv8 (primary)
# --------------------------------------------------------------------------- #


def ultralytics_available() -> bool:
    """Return True when ``ultralytics`` can be imported."""
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception as exc:  # ImportError or a broken install
        log.warning("Ultralytics unavailable (%s). Fallback SSD path will be used.", exc)
        return False


def build_yolo(cfg: Dict[str, Any], weights: Optional[str] = None):
    """Create a YOLOv8 model ready for training or inference.

    Args:
        cfg: Loaded configuration. Uses ``model_variant`` and ``pretrained``.
        weights: Explicit ``.pt`` path. When given it overrides the variant;
            use this to load ``best.pt`` after training.

    Returns:
        An ``ultralytics.YOLO`` instance.

    Raises:
        ValueError: If ``model_variant`` is not one of the supported sizes.
        FileNotFoundError: If ``weights`` is given but does not exist.
    """
    from ultralytics import YOLO

    if weights is not None:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Checkpoint {weights} not found. Train first (notebook 02) or "
                "fix the path in configs/default.yaml (save_dir / run_name)."
            )
        log.info("Loading YOLO weights from %s", weights)
        return YOLO(str(weights))

    variant = cfg["model_variant"]
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"model_variant must be one of {SUPPORTED_VARIANTS}, got '{variant}'")
    source = f"{variant}.pt" if cfg.get("pretrained", True) else f"{variant}.yaml"
    log.info("Building %s (%s)", variant, "COCO-pretrained" if cfg.get("pretrained", True) else "from scratch")
    return YOLO(source)


def load_trained_yolo(cfg: Dict[str, Any], weights: Optional[str] = None):
    """Load the best trained checkpoint (or an explicit path).

    Args:
        cfg: Loaded configuration.
        weights: Optional explicit path; defaults to ``<run_dir>/weights/best.pt``.

    Returns:
        An ``ultralytics.YOLO`` instance in eval mode.

    Raises:
        FileNotFoundError: With instructions when no checkpoint exists yet.
    """
    path = Path(weights) if weights else find_best_checkpoint(cfg)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "No trained checkpoint found. Expected best.pt under "
            f"{cfg['save_dir']}/{cfg['run_name']}/weights/. Run notebook 02 first."
        )
    return build_yolo(cfg, weights=str(path))


def model_summary(model: Any) -> Dict[str, Any]:
    """Return parameter count / GFLOPs for a YOLO model (safe on any object)."""
    try:
        from ultralytics.utils.torch_utils import model_info

        n_layers, n_params, n_grad, flops = model_info(model.model, detailed=False, verbose=False)
        return {"layers": n_layers, "parameters": n_params, "gradients": n_grad, "GFLOPs": flops}
    except Exception:  # pragma: no cover - depends on ultralytics internals
        n_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else -1
        return {"parameters": n_params}


# --------------------------------------------------------------------------- #
# SSD-MobileNetV2 (fallback)
# --------------------------------------------------------------------------- #


def _mobilenetv2_ssd_backbone(pretrained: bool = True):
    """MobileNetV2 feature extractor producing six SSD feature maps.

    Layout mirrors SSDLite-MobileNetV2: stride-16 (96 ch) and stride-32
    (1280 ch) maps from the trunk, then four extra stride-2 blocks.
    """
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    trunk = mobilenet_v2(weights=weights).features

    def extra_block(in_ch: int, out_ch: int) -> nn.Sequential:
        mid = out_ch // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True),
        )

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = trunk[:14]   # -> 96 ch, stride 16
            self.stage2 = trunk[14:]   # -> 1280 ch, stride 32
            self.extras = nn.ModuleList([
                extra_block(1280, 512), extra_block(512, 256),
                extra_block(256, 256), extra_block(256, 128),
            ])
            self.out_channels: List[int] = [96, 1280, 512, 256, 256, 128]

        def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
            feats: List[torch.Tensor] = []
            x = self.stage1(x)
            feats.append(x)
            x = self.stage2(x)
            feats.append(x)
            for block in self.extras:
                x = block(x)
                feats.append(x)
            return OrderedDict((str(i), f) for i, f in enumerate(feats))

    return Backbone()


def build_ssd_mobilenetv2(cfg: Dict[str, Any], pretrained_backbone: bool = True):
    """Assemble an SSD detector with a MobileNetV2 backbone (fallback model).

    Args:
        cfg: Loaded configuration (``num_classes``, ``image_size``,
            ``confidence_threshold``, ``iou_threshold``).
        pretrained_backbone: Initialise the trunk from ImageNet weights.

    Returns:
        A ``torchvision.models.detection.SSD`` module. Class ``0`` is
        background, so the head predicts ``num_classes + 1`` outputs; the
        dataset in :mod:`src.dataset` already offsets labels by +1.
    """
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.ssd import SSD

    backbone = _mobilenetv2_ssd_backbone(pretrained_backbone)
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]], min_ratio=0.2, max_ratio=0.95
    )
    size = int(cfg["image_size"])
    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(size, size),
        num_classes=int(cfg["num_classes"]) + 1,
        score_thresh=float(cfg["confidence_threshold"]),
        nms_thresh=float(cfg["iou_threshold"]),
        detections_per_img=int(cfg.get("max_detections", 300)),
    )
    log.info("Built SSD-MobileNetV2 fallback (%d classes + background)", cfg["num_classes"])
    return model


def build_model(cfg: Dict[str, Any], weights: Optional[str] = None, force_fallback: bool = False):
    """Return the primary YOLO model, or the SSD fallback if needed.

    Args:
        cfg: Loaded configuration.
        weights: Optional checkpoint path (YOLO only).
        force_fallback: Skip Ultralytics even if it imports (for testing).

    Returns:
        ``(model, backend)`` where backend is ``"yolo"`` or ``"ssd"``.
    """
    if not force_fallback and ultralytics_available():
        return build_yolo(cfg, weights), "yolo"
    return build_ssd_mobilenetv2(cfg), "ssd"
