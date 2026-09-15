"""Inference pipeline: single image, URL, video file and live frames.

Usage::

    from src.inference import Detector
    det = Detector(cfg)                      # loads best.pt from the run dir
    annotated, detections = det.detect_image(image_bgr)
    det.process_video("in.mp4", "out.mp4")

Every detection is returned as a :class:`Detection` and the annotated frame
carries colour-coded boxes, ``"class 0.94"`` labels and the summary banner.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils import draw_detections, ensure_dir, get_logger, summarise_counts

log = get_logger(__name__)


@dataclass
class Detection:
    """One detected face."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # absolute pixels [x1, y1, x2, y2]

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict (JSON serialisable)."""
        d = asdict(self)
        d["confidence"] = round(d["confidence"], 4)
        d["bbox_xyxy"] = [round(v, 1) for v in d["bbox_xyxy"]]
        return d


class Detector:
    """Thin, backend-agnostic wrapper around a trained detector.

    Args:
        cfg: Loaded configuration.
        weights: Optional explicit checkpoint path. Defaults to the run's
            ``best.pt`` for YOLO, ``<run>/weights/best.pt`` for SSD.
        backend: ``"yolo"`` (default) or ``"ssd"`` (fallback model).
        device: ``"0"`` / ``"cpu"`` / ``None`` (auto).
    """

    def __init__(self, cfg: Dict[str, Any], weights: Optional[str] = None,
                 backend: str = "yolo", device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self.names: List[str] = cfg["class_names"]
        self.conf = float(cfg["confidence_threshold"])
        self.iou = float(cfg["iou_threshold"])
        self.imgsz = int(cfg["image_size"])
        self.max_det = int(cfg.get("max_detections", 300))

        import torch

        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")

        if backend == "yolo":
            from src.model import load_trained_yolo

            self.model = load_trained_yolo(cfg, weights)
        elif backend == "ssd":
            from src.model import build_ssd_mobilenetv2
            from src.utils import run_dir

            path = Path(weights) if weights else run_dir(cfg) / "weights" / "best.pt"
            if not path.exists():
                raise FileNotFoundError(f"SSD checkpoint {path} not found. Train with --backend ssd first.")
            self.model = build_ssd_mobilenetv2(cfg, pretrained_backbone=False)
            state = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state["model"] if "model" in state else state)
            self.torch_device = torch.device("cuda:0" if self.device != "cpu" else "cpu")
            self.model.to(self.torch_device).eval()
        else:
            raise ValueError("backend must be 'yolo' or 'ssd'")
        log.info("Detector ready (backend=%s, device=%s)", backend, self.device)

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #

    def predict(self, image_bgr: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Run the model on one BGR image and return detections.

        Args:
            image_bgr: ``(H, W, 3)`` uint8 BGR array.
            conf: Optional confidence override.

        Returns:
            List of :class:`Detection` sorted by confidence (descending).
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to Detector.predict")
        conf = self.conf if conf is None else float(conf)
        if self.backend == "yolo":
            res = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, iou=self.iou,
                                     max_det=self.max_det, device=self.device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return []
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
        else:
            boxes, ids, confs = self._predict_ssd(image_bgr, conf)

        dets = [
            Detection(int(c), self.names[int(c)] if 0 <= int(c) < len(self.names) else str(c),
                      float(s), [float(v) for v in b])
            for b, c, s in zip(boxes, ids, confs)
        ]
        return sorted(dets, key=lambda d: -d.confidence)

    def _predict_ssd(self, image_bgr: np.ndarray, conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SSD fallback forward pass (letterbox-free resize, then rescale boxes)."""
        import torch

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.torch_device)
        with torch.no_grad():
            out = self.model([tensor])[0]
        keep = out["scores"] >= conf
        boxes = out["boxes"][keep].cpu().numpy()
        ids = out["labels"][keep].cpu().numpy().astype(int) - 1  # undo background offset
        confs = out["scores"][keep].cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, ids, confs

    def annotate(self, image_bgr: np.ndarray, detections: List[Detection], show_summary: bool = True) -> np.ndarray:
        """Draw detections on a copy of the image (BGR)."""
        return draw_detections(
            image_bgr,
            [d.bbox_xyxy for d in detections],
            [d.class_id for d in detections],
            [d.confidence for d in detections],
            self.cfg,
            show_summary=show_summary,
        )

    def detect_image(self, image_bgr: np.ndarray, conf: Optional[float] = None) -> Tuple[np.ndarray, List[Detection]]:
        """Predict + annotate in one call. Returns ``(annotated_bgr, detections)``."""
        dets = self.predict(image_bgr, conf)
        return self.annotate(image_bgr, dets), dets

    def summary(self, detections: List[Detection]) -> Dict[str, int]:
        """Per-class counts plus ``total`` for a list of detections."""
        return summarise_counts([d.class_id for d in detections], self.cfg)

    # ------------------------------------------------------------------ #
    # Convenience inputs
    # ------------------------------------------------------------------ #

    def detect_file(self, path: "str | Path", save_to: Optional["str | Path"] = None) -> Tuple[np.ndarray, List[Detection]]:
        """Detect on an image file; optionally write the annotated result.

        Raises:
            FileNotFoundError: If the file is missing or unreadable.
        """
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image {path}. Is it a valid JPG/PNG?")
        annotated, dets = self.detect_image(img)
        if save_to:
            ensure_dir(Path(save_to).parent)
            cv2.imwrite(str(save_to), annotated)
        return annotated, dets

    def detect_url(self, url: str, timeout: int = 20) -> Tuple[np.ndarray, List[Detection]]:
        """Download an image from a URL and detect on it.

        Raises:
            RuntimeError: If the download fails or the payload is not an image.
        """
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = np.frombuffer(resp.read(), dtype=np.uint8)
        except Exception as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"URL {url} did not return a decodable image.")
        return self.detect_image(img)

    def process_video(
        self,
        input_path: "str | Path",
        output_path: "str | Path",
        every_n: int = 1,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Annotate a video frame-by-frame and write it to ``output_path``.

        Args:
            input_path: Source ``.mp4`` / ``.avi`` etc.
            output_path: Destination ``.mp4`` (parent dirs are created).
            every_n: Run the model every N frames and reuse the last boxes in
                between (speeds up long videos; 1 = every frame).
            max_frames: Stop early (useful for quick tests).
            show_progress: Display a tqdm bar.

        Returns:
            Summary dict: frames processed, FPS, per-class totals, output path.

        Raises:
            FileNotFoundError: If the input cannot be opened.
        """
        from tqdm.auto import tqdm

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {input_path}. Check the path / codec.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if max_frames:
            total = min(total or max_frames, max_frames)

        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video writer for {output_path}.")

        totals = {n: 0 for n in self.names}
        dets: List[Detection] = []
        n_done, t0 = 0, time.perf_counter()
        bar = tqdm(total=total, desc="video", disable=not show_progress)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames and n_done >= max_frames):
                    break
                if n_done % max(1, every_n) == 0:
                    dets = self.predict(frame)
                    for d in dets:
                        totals[d.class_name] += 1
                writer.write(self.annotate(frame, dets))
                n_done += 1
                bar.update(1)
        finally:
            bar.close()
            cap.release()
            writer.release()
        elapsed = time.perf_counter() - t0
        summary = {
            "frames": n_done, "seconds": round(elapsed, 2),
            "fps": round(n_done / max(elapsed, 1e-9), 1),
            "detections_per_class": totals, "output": str(output_path),
        }
        log.info("Video done: %s", json.dumps(summary))
        return summary

    def detect_frame_jpeg(self, jpeg_bytes: bytes) -> Tuple[bytes, List[Detection]]:
        """Decode a JPEG (e.g. from the Colab webcam JS), detect, re-encode.

        Returns:
            ``(annotated_jpeg_bytes, detections)``.
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode JPEG bytes from the webcam capture.")
        annotated, dets = self.detect_image(frame)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG encoding of the annotated frame failed.")
        return buf.tobytes(), dets


def detections_to_json(detections: List[Detection], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the JSON payload used by the Gradio demo and notebook 04."""
    return {
        "summary": summarise_counts([d.class_id for d in detections], cfg),
        "detections": [d.to_dict() for d in detections],
    }
