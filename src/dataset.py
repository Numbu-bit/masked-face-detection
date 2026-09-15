"""Dataset preparation, augmentation and loading.

Responsibilities
----------------
1. Convert the Kaggle ``andrewmvd/face-mask-detection`` Pascal-VOC XML labels
   into YOLO ``.txt`` files (``class x_center y_center w h``, normalised).
2. Ingest a Roboflow YOLOv8 export (already in YOLO format).
3. Perform a *stratified* 70/15/15 train/val/test split and lay files out in
   the ``images/`` + ``labels/`` structure Ultralytics expects.
4. Write ``data.yaml``.
5. Provide an Albumentations pipeline (with YOLO bbox params) and a PyTorch
   :class:`MaskDataset` used by the SSD fallback and for visual sanity checks.

All paths and ratios come from ``configs/default.yaml``.
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from src.utils import ensure_dir, get_logger

log = get_logger(__name__)

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS: Tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    """One image plus its YOLO-format labels (in memory)."""

    image_path: Path
    labels: List[Tuple[int, float, float, float, float]] = field(default_factory=list)

    @property
    def dominant_class(self) -> int:
        """Most frequent class in the image; -1 when there are no boxes.

        Used as the stratification key so each split has a similar class mix.
        """
        if not self.labels:
            return -1
        return Counter(lbl[0] for lbl in self.labels).most_common(1)[0][0]


# --------------------------------------------------------------------------- #
# Pascal-VOC (Kaggle) -> YOLO
# --------------------------------------------------------------------------- #


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def voc_box_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert absolute VOC corners to normalised YOLO centre/size.

    Args:
        xmin, ymin, xmax, ymax: Absolute pixel corners.
        img_w, img_h: Image width / height in pixels.

    Returns:
        ``(x_center, y_center, width, height)`` each in ``[0, 1]``.
    """
    xc = _clip01(((xmin + xmax) / 2.0) / img_w)
    yc = _clip01(((ymin + ymax) / 2.0) / img_h)
    w = _clip01((xmax - xmin) / img_w)
    h = _clip01((ymax - ymin) / img_h)
    return xc, yc, w, h


def sanitize_yolo_box(xc: float, yc: float, w: float, h: float, eps: float = 1e-6) -> List[float]:
    """Clip a normalised YOLO box so its corners lie strictly inside ``[0, 1]``.

    Albumentations validates ``x_min = xc - w/2 >= 0`` etc. and rejects boxes
    that are off by floating-point noise (e.g. ``-5e-7``). We convert to
    corners, clip, and convert back.
    """
    x1 = min(max(xc - w / 2, 0.0), 1.0 - eps)
    y1 = min(max(yc - h / 2, 0.0), 1.0 - eps)
    x2 = min(max(xc + w / 2, x1 + eps), 1.0)
    y2 = min(max(yc + h / 2, y1 + eps), 1.0)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def parse_voc_xml(xml_path: Path, class_map: Dict[str, str], class_names: Sequence[str]) -> Tuple[Optional[str], List[Tuple[int, float, float, float, float]]]:
    """Parse one Pascal-VOC annotation file into YOLO labels.

    Args:
        xml_path: Path to the ``.xml`` file.
        class_map: Raw VOC class name -> canonical class name.
        class_names: Canonical ordered class list (index = class id).

    Returns:
        ``(image_filename, labels)``. ``image_filename`` is ``None`` when the
        XML is unreadable; unknown classes are skipped with a warning.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        log.warning("Skipping malformed XML %s: %s", xml_path.name, exc)
        return None, []

    filename = root.findtext("filename") or xml_path.with_suffix(".png").name
    size = root.find("size")
    if size is None:
        log.warning("No <size> in %s; skipping", xml_path.name)
        return None, []
    img_w = int(float(size.findtext("width", "0")))
    img_h = int(float(size.findtext("height", "0")))
    if img_w <= 0 or img_h <= 0:
        log.warning("Invalid image size in %s; skipping", xml_path.name)
        return None, []

    labels: List[Tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        canonical = class_map.get(raw_name)
        if canonical is None or canonical not in class_names:
            log.debug("Unknown class '%s' in %s; skipped", raw_name, xml_path.name)
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        cid = list(class_names).index(canonical)
        labels.append((cid, *voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)))
    return filename, labels


def collect_voc_samples(raw_dir: Path, cfg: Dict[str, Any]) -> List[Sample]:
    """Walk a Kaggle VOC layout (``images/`` + ``annotations/``) into Samples.

    Args:
        raw_dir: Directory containing ``images`` and ``annotations`` folders.
        cfg: Loaded configuration (uses ``voc_class_map`` and ``class_names``).

    Returns:
        List of :class:`Sample` whose image file actually exists.

    Raises:
        FileNotFoundError: If the expected folders are missing.
    """
    img_dir, ann_dir = raw_dir / "images", raw_dir / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {raw_dir}/images and {raw_dir}/annotations. "
            "Check that the Kaggle download finished and unzipped correctly."
        )
    class_map: Dict[str, str] = cfg["voc_class_map"]
    names: List[str] = cfg["class_names"]

    samples: List[Sample] = []
    missing = 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        filename, labels = parse_voc_xml(xml_path, class_map, names)
        if filename is None:
            continue
        image_path = img_dir / filename
        if not image_path.exists():  # try alternative extensions
            candidates = [img_dir / (xml_path.stem + ext) for ext in IMAGE_EXTS]
            image_path = next((c for c in candidates if c.exists()), image_path)
        if not image_path.exists():
            missing += 1
            continue
        samples.append(Sample(image_path=image_path, labels=labels))
    if missing:
        log.warning("%d annotations had no matching image and were skipped", missing)
    log.info("Collected %d VOC samples from %s", len(samples), raw_dir)
    return samples


# --------------------------------------------------------------------------- #
# Roboflow YOLOv8 export -> Samples
# --------------------------------------------------------------------------- #


def read_yolo_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read a YOLO ``.txt`` label file; missing file -> empty list (background)."""
    if not path.exists():
        return []
    labels: List[Tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        labels.append((cid, _clip01(xc), _clip01(yc), _clip01(w), _clip01(h)))
    return labels


def collect_yolo_samples(root: Path, remap: Optional[Dict[int, int]] = None) -> List[Sample]:
    """Collect samples from any tree containing ``images/`` + ``labels/`` pairs.

    Works for Roboflow exports (``train/images``, ``valid/images`` ...) and for
    our own prepared layout.

    Args:
        root: Directory to search recursively.
        remap: Optional ``{source_class_id: target_class_id}`` mapping used when
            the source dataset's class order differs from ``cfg["class_names"]``.

    Returns:
        List of :class:`Sample`.
    """
    samples: List[Sample] = []
    for img_dir in sorted(p for p in root.rglob("images") if p.is_dir()):
        lbl_dir = img_dir.parent / "labels"
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            labels = read_yolo_label_file(lbl_dir / (img_path.stem + ".txt"))
            if remap:
                labels = [(remap.get(l[0], l[0]), *l[1:]) for l in labels]
            samples.append(Sample(img_path, labels))
    log.info("Collected %d YOLO samples under %s", len(samples), root)
    return samples


def read_roboflow_names(data_yaml: Path) -> List[str]:
    """Return the ``names`` list from a Roboflow ``data.yaml``."""
    with open(data_yaml, "r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}
    names = meta.get("names", [])
    if isinstance(names, dict):  # {0: 'a', 1: 'b'} form
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def build_class_remap(source_names: Sequence[str], target_names: Sequence[str]) -> Dict[int, int]:
    """Map source class ids to target ids by (normalised) name.

    Names are compared case-insensitively with ``-``/`` `` replaced by ``_`` so
    ``"Mask Worn Incorrectly"`` matches ``"mask_worn_incorrectly"``.
    A few common aliases are handled explicitly.

    Raises:
        ValueError: If a source class has no counterpart in ``target_names``.
    """
    aliases = {
        "mask": "with_mask",
        "no_mask": "without_mask",
        "no-mask": "without_mask",
        "nomask": "without_mask",
        "mask_weared_incorrect": "mask_worn_incorrectly",
        "incorrect_mask": "mask_worn_incorrectly",
        "improperly_worn": "mask_worn_incorrectly",
    }

    def norm(n: str) -> str:
        n = n.strip().lower().replace("-", "_").replace(" ", "_")
        return aliases.get(n, n)

    target_idx = {norm(n): i for i, n in enumerate(target_names)}
    remap: Dict[int, int] = {}
    for i, src in enumerate(source_names):
        key = norm(src)
        if key not in target_idx:
            raise ValueError(
                f"Source class '{src}' cannot be mapped onto {list(target_names)}. "
                "Add an alias in build_class_remap() or edit class_names in the config."
            )
        remap[i] = target_idx[key]
    return remap


# --------------------------------------------------------------------------- #
# Stratified split + on-disk layout
# --------------------------------------------------------------------------- #


def stratified_split(
    samples: Sequence[Sample], ratios: Sequence[float], seed: int
) -> Dict[str, List[Sample]]:
    """Split samples 70/15/15 (or as configured) stratified by dominant class.

    Args:
        samples: All samples.
        ratios: ``[train, val, test]`` fractions summing to 1.
        seed: RNG seed for reproducibility.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``.

    Raises:
        ValueError: If ratios are invalid or there are no samples.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must be three numbers summing to 1, got {ratios}")
    if not samples:
        raise ValueError("No samples to split. Did the download / conversion step succeed?")

    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}
    for s in samples:
        by_class.setdefault(s.dominant_class, []).append(s)

    out: Dict[str, List[Sample]] = {k: [] for k in SPLITS}
    for _, group in sorted(by_class.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        out["train"] += group[:n_train]
        out["val"] += group[n_train : n_train + n_val]
        out["test"] += group[n_train + n_val :]
    for k in SPLITS:
        rng.shuffle(out[k])
    log.info("Split -> train=%d val=%d test=%d", *(len(out[k]) for k in SPLITS))
    return out


def write_split(
    split: Dict[str, List[Sample]],
    out_root: Path,
    image_size: Optional[int] = None,
    copy: bool = True,
) -> Path:
    """Write ``images/`` and ``labels/`` folders for each split.

    Args:
        split: Output of :func:`stratified_split`.
        out_root: Destination (e.g. ``/content/data``). Existing split folders
            are removed first so re-running the notebook is idempotent.
        image_size: If given, images are resized (letterbox-free, aspect kept
            by Ultralytics at train time) to at most this size on the long
            side. This shrinks Drive usage; labels are normalised so remain valid.
        copy: Copy files (True) or move them (False).

    Returns:
        ``out_root``.
    """
    out_root = ensure_dir(out_root)
    for name in SPLITS:
        for sub in ("images", "labels"):
            d = out_root / name / sub
            if d.exists():
                shutil.rmtree(d)
            ensure_dir(d)

    from tqdm.auto import tqdm

    for name in SPLITS:
        for s in tqdm(split[name], desc=f"writing {name}", leave=False):
            dst_img = out_root / name / "images" / (s.image_path.stem + ".jpg")
            dst_lbl = out_root / name / "labels" / (s.image_path.stem + ".txt")
            if image_size is not None or s.image_path.suffix.lower() != ".jpg":
                img = cv2.imread(str(s.image_path))
                if img is None:
                    log.warning("Unreadable image %s; skipped", s.image_path)
                    continue
                if image_size is not None:
                    h, w = img.shape[:2]
                    scale = image_size / max(h, w)
                    if scale < 1.0:
                        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif copy:
                shutil.copy2(s.image_path, dst_img)
            else:
                shutil.move(str(s.image_path), dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as fh:
                for cid, xc, yc, w, h in s.labels:
                    fh.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return out_root


def write_data_yaml(data_root: Path, class_names: Sequence[str], path: Optional[Path] = None) -> Path:
    """Write the ``data.yaml`` Ultralytics needs.

    Args:
        data_root: Directory holding ``train/ val/ test/``.
        class_names: Ordered class names.
        path: Output path; defaults to ``data_root / "data.yaml"``.

    Returns:
        Path to the written file.
    """
    path = path or data_root / "data.yaml"
    meta = {
        "path": str(data_root),
        "train": str(data_root / "train" / "images"),
        "val": str(data_root / "val" / "images"),
        "test": str(data_root / "test" / "images"),
        "nc": len(class_names),
        "names": list(class_names),
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    log.info("Wrote %s", path)
    return path


def dataset_statistics(data_root: Path, class_names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Count images and per-class boxes in each split.

    Returns:
        ``{"train": {"images": n, "with_mask": k, ...}, "val": ..., "test": ...}``
    """
    stats: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        lbl_dir = data_root / split / "labels"
        img_dir = data_root / split / "images"
        counts = Counter()
        n_imgs = len([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]) if img_dir.exists() else 0
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                for row in read_yolo_label_file(lbl):
                    counts[row[0]] += 1
        stats[split] = {"images": n_imgs, **{name: counts.get(i, 0) for i, name in enumerate(class_names)}}
    return stats


def validate_yolo_dataset(data_root: Path, num_classes: int) -> List[str]:
    """Return a list of problems found (empty list == dataset is clean).

    Checks: every image has a label file, class ids in range, coords in [0, 1].
    """
    problems: List[str] = []
    for split in SPLITS:
        img_dir, lbl_dir = data_root / split / "images", data_root / split / "labels"
        if not img_dir.exists():
            problems.append(f"missing {img_dir}")
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                problems.append(f"{split}: no label for {img.name}")
                continue
            for i, row in enumerate(read_yolo_label_file(lbl)):
                if not 0 <= row[0] < num_classes:
                    problems.append(f"{split}: {lbl.name} line {i} class {row[0]} out of range")
                if any(not 0.0 <= v <= 1.0 for v in row[1:]):
                    problems.append(f"{split}: {lbl.name} line {i} coords outside [0,1]")
    return problems


# --------------------------------------------------------------------------- #
# Albumentations
# --------------------------------------------------------------------------- #


def _bbox_safe_coarse_dropout():
    """Return a CoarseDropout subclass that leaves bounding boxes untouched.

    Cutout only blanks pixels, so boxes are unchanged - but Albumentations
    1.4.4 raises ``NotImplementedError`` when the transform sees bboxes.
    """
    import albumentations as A

    class CoarseDropoutBBoxSafe(A.CoarseDropout):  # type: ignore[misc]
        def apply_to_bbox(self, bbox, **params):  # noqa: D401 - identity
            return bbox

        def apply_to_bboxes(self, bboxes, **params):
            return bboxes

    return CoarseDropoutBBoxSafe


def build_train_augmentations(cfg: Dict[str, Any]):
    """Build the Albumentations training pipeline described in the spec.

    Args:
        cfg: Loaded configuration; reads ``cfg["albumentations"]``.

    Returns:
        An ``albumentations.Compose`` with ``BboxParams(format="yolo")``.
    """
    import albumentations as A

    a = cfg["albumentations"]
    CoarseDropout = _bbox_safe_coarse_dropout()
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=a["random_brightness_contrast_p"]),
            A.HorizontalFlip(p=a["horizontal_flip_p"]),
            A.RandomScale(scale_limit=a["random_scale_limit"], p=0.5),
            A.GaussianBlur(blur_limit=(3, a["gaussian_blur_limit"]), p=a["gaussian_blur_p"]),
            A.CLAHE(p=a["clahe_p"]),
            A.ColorJitter(p=a["color_jitter_p"]),
            CoarseDropout(
                max_holes=a["coarse_dropout_max_holes"],
                max_height=32,
                max_width=32,
                fill_value=0,
                p=a["coarse_dropout_p"],
            ),
            A.LongestMaxSize(max_size=cfg["image_size"]),
            A.PadIfNeeded(cfg["image_size"], cfg["image_size"], border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def build_eval_transforms(cfg: Dict[str, Any]):
    """Deterministic resize/pad used for validation and the SSD fallback."""
    import albumentations as A

    return A.Compose(
        [
            A.LongestMaxSize(max_size=cfg["image_size"]),
            A.PadIfNeeded(cfg["image_size"], cfg["image_size"], border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


# --------------------------------------------------------------------------- #
# PyTorch Dataset (SSD fallback + visual sanity checks)
# --------------------------------------------------------------------------- #


class MaskDataset:
    """Minimal ``torch.utils.data.Dataset`` over a prepared YOLO split.

    Ultralytics builds its own loaders, so this class is used by the SSD
    fallback (``torchvision`` expects absolute ``xyxy`` targets) and by the
    notebooks to visualise augmentations.

    Args:
        data_root: Directory containing ``<split>/images`` and ``<split>/labels``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        cfg: Loaded configuration.
        augment: Apply the training augmentation pipeline.
    """

    def __init__(self, data_root: Path, split: str, cfg: Dict[str, Any], augment: bool = False) -> None:
        self.img_dir = Path(data_root) / split / "images"
        self.lbl_dir = Path(data_root) / split / "labels"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"{self.img_dir} does not exist. Run notebook 01 first.")
        self.images = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        self.cfg = cfg
        self.transform = build_train_augmentations(cfg) if augment else build_eval_transforms(cfg)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """Return ``(image_tensor[3,H,W] float 0-1, target)``.

        ``target`` has ``boxes`` (absolute xyxy, float tensor), ``labels``
        (int64 tensor, +1 so that 0 stays "background" for torchvision) and
        ``image_id``.
        """
        import torch

        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Could not read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows = read_yolo_label_file(self.lbl_dir / (img_path.stem + ".txt"))
        bboxes = [sanitize_yolo_box(*r[1:]) for r in rows]
        class_labels = [r[0] for r in rows]

        out = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        image, bboxes, class_labels = out["image"], out["bboxes"], out["class_labels"]

        h, w = image.shape[:2]
        boxes_xyxy = []
        for xc, yc, bw, bh in bboxes:
            boxes_xyxy.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        target = {
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([c + 1 for c in class_labels], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor, target


def collate_detection(batch: Iterable[Tuple[Any, Dict[str, Any]]]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Collate function for variable-length detection targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_dataloader(data_root: Path, split: str, cfg: Dict[str, Any], augment: bool = False, shuffle: Optional[bool] = None):
    """Create a ``DataLoader`` over :class:`MaskDataset` using config values."""
    from torch.utils.data import DataLoader

    ds = MaskDataset(data_root, split, cfg, augment=augment)
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=cfg["num_workers"],
        collate_fn=collate_detection,
        pin_memory=True,
    )


# --------------------------------------------------------------------------- #
# Visual sanity check helper
# --------------------------------------------------------------------------- #


def yolo_to_xyxy(labels: Sequence[Tuple[int, float, float, float, float]], w: int, h: int) -> Tuple[List[List[float]], List[int]]:
    """Convert normalised YOLO rows to absolute ``xyxy`` boxes + class ids."""
    boxes, ids = [], []
    for cid, xc, yc, bw, bh in labels:
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        ids.append(cid)
    return boxes, ids


def load_annotated_sample(data_root: Path, split: str, stem: str, cfg: Dict[str, Any]) -> np.ndarray:
    """Read an image from a split and draw its ground-truth boxes (BGR)."""
    from src.utils import draw_detections

    img = cv2.imread(str(data_root / split / "images" / f"{stem}.jpg"))
    if img is None:
        raise FileNotFoundError(f"{stem}.jpg not found in {split}/images")
    rows = read_yolo_label_file(data_root / split / "labels" / f"{stem}.txt")
    boxes, ids = yolo_to_xyxy(rows, img.shape[1], img.shape[0])
    return draw_detections(img, boxes, ids, [1.0] * len(ids), cfg, show_summary=False)
