"""Gradio web UI for the masked-face detector (runs inside Colab with a public link).

Run from the repo root::

    python demo/gradio_app.py                 # local URL only
    python demo/gradio_app.py --share         # public *.gradio.live link (Colab)
    python demo/gradio_app.py --weights /content/drive/MyDrive/.../best.pt

Or from a notebook cell::

    from demo.gradio_app import build_demo
    build_demo(cfg).launch(share=True)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import Detector, detections_to_json  # noqa: E402
from src.utils import bgr_to_rgb, get_logger, load_config, rgb_to_bgr  # noqa: E402

log = get_logger("gradio")

EXAMPLES_DIR = ROOT / "demo" / "examples"


def _example_images(cfg: Dict[str, Any], n: int = 3) -> List[List[str]]:
    """Return ``[[path], ...]`` example images for the Gradio gallery.

    Uses ``demo/examples/`` when it contains images; otherwise copies ``n``
    images from the prepared test split into it (so the demo always has
    examples once notebook 01 has run).
    """
    import shutil

    exts = (".jpg", ".jpeg", ".png")
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    found = [p for p in sorted(EXAMPLES_DIR.iterdir()) if p.suffix.lower() in exts]
    if not found:
        test_dir = Path(cfg["data_root"]) / "test" / "images"
        if test_dir.exists():
            for p in sorted(p for p in test_dir.iterdir() if p.suffix.lower() in exts)[:n]:
                dst = EXAMPLES_DIR / f"example_{p.name}"
                shutil.copy2(p, dst)
                found.append(dst)
            log.info("Copied %d example images from %s", len(found), test_dir)
    return [[str(p)] for p in found]


def build_demo(cfg: Dict[str, Any], weights: Optional[str] = None):
    """Create (but do not launch) the Gradio app.

    Two tabs:

    * **Image** - upload a file *or* take a webcam snapshot, get the annotated
      image + JSON details.
    * **Live webcam** - the browser streams frames and each one is annotated
      in near real-time (the browser asks for camera permission).

    Args:
        cfg: Loaded configuration.
        weights: Optional checkpoint path; defaults to the run's ``best.pt``.

    Returns:
        A ``gradio.Blocks`` (``TabbedInterface``).
    """
    import gradio as gr

    detector = Detector(cfg, weights=weights)
    major = int(gr.__version__.split(".")[0])
    # gradio 5 renamed allow_flagging -> flagging_mode
    flag_kw = {"flagging_mode": "never"} if major >= 5 else {"allow_flagging": "never"}

    def detect_faces(image: np.ndarray, conf: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gradio callback: RGB numpy in -> annotated RGB + JSON details out."""
        if image is None:
            raise gr.Error("Please upload an image or take a webcam snapshot first.")
        try:
            annotated_bgr, dets = detector.detect_image(rgb_to_bgr(image), conf=conf)
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            raise gr.Error(f"Detection failed: {exc}") from exc
        return bgr_to_rgb(annotated_bgr), detections_to_json(dets, cfg)

    def detect_stream(frame: np.ndarray, conf: float) -> np.ndarray:
        """Streaming callback: one webcam frame in -> annotated frame out (no JSON, for speed)."""
        if frame is None:
            return None
        annotated_bgr, _ = detector.detect_image(rgb_to_bgr(frame), conf=conf)
        return bgr_to_rgb(annotated_bgr)

    legend = " · ".join(
        f"<span style='color:rgb({c[2]},{c[1]},{c[0]});font-weight:bold'>{name}</span>"
        for name, c in cfg["class_colors"].items()
    )
    description = (
        "Detects faces and classifies each one as **mask on**, **mask off**, or "
        f"**mask worn incorrectly**. Legend: {legend}. "
        f"Model: {cfg['model_variant']} · input {cfg['image_size']}px."
    )
    conf_slider = lambda: gr.Slider(  # noqa: E731 - tiny factory, one per tab
        0.05, 0.95, value=float(cfg["confidence_threshold"]), step=0.05, label="Confidence threshold"
    )

    image_tab = gr.Interface(
        fn=detect_faces,
        inputs=[
            gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Upload an image or take a webcam snapshot"),
            conf_slider(),
        ],
        outputs=[gr.Image(label="Detection Result"), gr.JSON(label="Detection Details")],
        title="Masked Face Detection",
        description=description,
        examples=_example_images(cfg) or None,
        cache_examples=False,
        **flag_kw,
    )

    webcam_tab = gr.Interface(
        fn=detect_stream,
        inputs=[gr.Image(type="numpy", sources=["webcam"], streaming=True, label="Webcam (live)"), conf_slider()],
        outputs=gr.Image(label="Live detection"),
        live=True,
        title="Masked Face Detection — live webcam",
        description="Allow camera access when the browser asks. Frames are sent to the model continuously; "
                    "expect a few FPS through a Colab share link, more when running locally. " + description,
        **flag_kw,
    )

    app = gr.TabbedInterface([image_tab, webcam_tab], tab_names=["Image / snapshot", "Live webcam"],
                             title="Masked Face Detection")
    # Expose the callbacks for programmatic use / tests (scripts/smoke_test.py).
    app.detect_faces = detect_faces
    app.detect_stream = detect_stream
    return app


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Launch the Gradio demo")
    parser.add_argument("--weights", default=None, help="Path to best.pt (default: run dir)")
    parser.add_argument("--config", default=None, help="Config YAML")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = load_config(args.config)
    demo = build_demo(cfg, weights=args.weights)
    demo.launch(share=args.share, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
