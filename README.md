# Masked Face Detection

Detects faces in images, video and live webcam streams and classifies each one as
**wearing a mask**, **not wearing a mask**, or **wearing a mask incorrectly**.
Built on YOLOv8 (Ultralytics / PyTorch) and designed to be trained and run entirely on
**Google Colab** (free T4 tier is enough) — every notebook mounts Google Drive, checkpoints
there, and resumes automatically after a disconnect.

![sample result](assets/sample_result.jpg)
*Placeholder — after running notebook 03, copy `runs/<run_name>/eval/prediction_grid_test.png` to `assets/sample_result.jpg`.*

---

## Classes

| id | class                     | colour                 | meaning                                            |
|----|---------------------------|------------------------|----------------------------------------------------|
| 0  | `with_mask`               | 🟩 green `(0,200,0)`   | nose and mouth covered                             |
| 1  | `without_mask`            | 🟥 red `(0,0,220)`     | no mask                                            |
| 2  | `mask_worn_incorrectly`   | 🟧 orange `(0,140,255)`| mask present but nose or mouth exposed             |

Colours are BGR (OpenCV) and defined once in `configs/default.yaml → class_colors`.
Every annotated frame also carries a banner: `Faces: 5 | Masked: 3 | Unmasked: 2 | Incorrect: 0`.

---

## Repository layout

```
masked-face-detection/
├── README.md
├── requirements.txt                  # pinned versions
├── configs/default.yaml              # ALL hyper-parameters, paths, thresholds
├── notebooks/
│   ├── 01_data_preparation.ipynb     # download → VOC/YOLO convert → stratified split → data.yaml → Drive backup
│   ├── 02_model_training.ipynb       # YOLOv8 fine-tuning, checkpoints on Drive, auto-resume
│   ├── 03_evaluation.ipynb           # mAP, P/R/F1, confusion matrix, PR curves, FPS, failure cases
│   └── 04_inference_demo.ipynb       # image / video / webcam inference, Gradio app, export
├── src/
│   ├── utils.py                      # config, seeding, device, drawing, checkpoint lookup
│   ├── dataset.py                    # VOC→YOLO, Roboflow ingestion, split, Albumentations, Dataset
│   ├── model.py                      # YOLOv8 builder + SSD-MobileNetV2 fallback
│   ├── train.py                      # train_yolo (auto-resume) / train_ssd + CLI
│   ├── evaluate.py                   # metrics, plots, failure analysis, FPS benchmark
│   └── inference.py                  # Detector: image / URL / video / JPEG frame
├── scripts/
│   ├── export_model.py               # ONNX / TFLite / TorchScript + validation vs. PyTorch
│   └── smoke_test.py                 # 3-minute CPU end-to-end self-test on synthetic data
└── demo/
    ├── gradio_app.py                 # Gradio web UI (public share link in Colab)
    └── examples/                     # sample images for the demo (auto-filled from the test split)
```

---

## Quick start on Google Colab

1. **Fork / push this repo to GitHub** (or copy the folder to `MyDrive/masked-face-detection/repo`).
2. Open a notebook in Colab: *File → Open notebook → GitHub → paste the notebook URL*.
3. *Runtime → Change runtime type → **T4 GPU***.
4. In the second cell of each notebook set `REPO_URL` to your fork (skip if you used the Drive copy).
5. Run the notebooks **in order**: `01 → 02 → 03 → 04`. Each starts with the same four cells
   (GPU check, Drive mount + repo, `pip install -r requirements.txt`, seed + config).

Everything persistent goes to `MyDrive/masked-face-detection/`:

```
MyDrive/masked-face-detection/
├── data_prepared.zip           # split dataset (notebook 01) — restored automatically by 02/03/04
├── runs/<run_name>/
│   ├── weights/{best,last,epochN}.pt   + best.onnx / best.torchscript / best_saved_model/
│   ├── results.csv, results.png, confusion_matrix.png …   (Ultralytics)
│   └── eval/                   # metrics_test.json + every figure from notebook 03
└── inference_outputs/          # annotated images / videos from notebook 04
```

### Providing credentials

| Source | What you need | Where to put it |
|---|---|---|
| **Kaggle (Option A, default)** | `kaggle.json` from *kaggle.com → Settings → API → Create New Token* | `MyDrive/kaggle.json`. If missing, notebook 01 prompts you to upload it and copies it to Drive for next time. |
| **Roboflow (Option B)** | API key from *Roboflow → Settings → API Keys* | Paste into the (commented) Option B cell together with the workspace / project / version you picked on universe.roboflow.com. |

Only one option is needed. Credentials never leave your Drive and are git-ignored.

### Datasets

| Priority | Dataset | Size | Notes |
|---|---|---|---|
| 1 | Kaggle [`andrewmvd/face-mask-detection`](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) | 853 images / ~4 000 faces, Pascal-VOC XML | 3 classes; `mask_weared_incorrect` is rare (~120 boxes) — expect that class to have the lowest F1 |
| 2 | Roboflow Universe "face mask detection" | varies (2–10 K images) | YOLOv8 export; class names are remapped automatically (`build_class_remap`) |
| 3 | WIDER FACE + MAFA | 30 K+ | no mask labels out-of-the-box; use only if you plan to relabel |

Notebook 01 converts any of these into the same layout: 70 / 15 / 15 stratified split, images
resized to ≤ 640 px, YOLO `.txt` labels, `data.yaml`.

---

## Running each notebook

| Notebook | What happens | Typical time (T4) |
|---|---|---|
| **01 data preparation** | download → explore (class histogram, GT samples) → convert & split → validate → preview Albumentations → zip to Drive | 3–5 min |
| **02 training** | optional 1-epoch smoke test → full training (AdamW, cosine LR, mosaic/mixup, early stop) → curves → quick val | 1–3 h for 100 epochs on the Kaggle set (`yolov8s`, 640 px) |
| **03 evaluation** | mAP@0.5 / 0.5:0.95, per-class P/R/F1, confusion matrix, PR curves, GPU+CPU FPS, 16-image grid, failure cases, target check with suggestions | 2–3 min |
| **04 inference demo** | upload/URL image → video file → Colab webcam (snapshot + live) → Gradio public link → export | interactive |

**Disconnected mid-training?** Reopen notebook 02, run the cells top to bottom. The training cell
finds `weights/last.pt` on Drive and resumes from that epoch. If the run had already finished it
says so and does nothing.

**CUDA out of memory?** In notebook 02 set `OVERRIDES = {"batch_size": 8}` (or `"image_size": 416`).

**Pro tier (optional):** `OVERRIDES = {"model_variant": "yolov8m", "batch_size": 32}` on an A100/L4.

---

## Configuration

All knobs live in [`configs/default.yaml`](configs/default.yaml). Highlights:

```yaml
model_variant: yolov8s      # yolov8n | yolov8s | yolov8m
image_size: 640             # 416 on tight memory
batch_size: 16
epochs: 100
optimizer: AdamW            # lr 0.001, wd 0.0005, cosine schedule, 3 warm-up epochs
patience: 15                # early stopping
mosaic: 1.0  mixup: 0.1     # YOLO-side augmentation
albumentations: …           # brightness/contrast, flip, scale, blur, CLAHE, colour-jitter, coarse dropout
confidence_threshold: 0.45
iou_threshold: 0.5
targets: {map50: 0.85, per_class_f1: 0.80, fps_gpu: 30}
```

Override anything per-session without editing the file: `load_config(overrides={"epochs": 50})`,
the `OVERRIDES` dict in notebook 02, or the `MFD_OVERRIDES` environment variable (JSON).

---

## Model performance

Fill this table from `runs/<run_name>/eval/metrics_test.json` after notebook 03
(the notebook prints a ready-to-paste JSON summary in its last cell).

| Model | img | mAP@0.5 | mAP@0.5:0.95 | F1 with_mask | F1 without_mask | F1 incorrect | FPS T4 | FPS CPU |
|---|---|---|---|---|---|---|---|---|
| yolov8s (default) | 640 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| yolov8n | 640 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Targets: mAP@0.5 ≥ 0.85, every class F1 ≥ 0.80, ≥ 30 FPS on T4. If any target is missed,
notebook 03 prints concrete suggestions (longer training, larger variant, more data,
oversampling the rare class, lower mosaic/mixup, smaller input for speed).

---

## Command-line usage (outside the notebooks)

```bash
pip install -r requirements.txt

# self-test on synthetic data (CPU, ~3 min) — run this first after any change
python scripts/smoke_test.py

# train (auto-resumes) / fallback SSD
python -m src.train --data /content/data/data.yaml
python -m src.train --data /content/data --backend ssd

# Gradio demo
python demo/gradio_app.py --share --weights path/to/best.pt

# export + validation
python scripts/export_model.py --weights path/to/best.pt --data-root /content/data
python scripts/export_model.py --formats onnx torchscript      # skip the TensorFlow install
```

Python API:

```python
from src.utils import load_config
from src.inference import Detector

cfg = load_config()
det = Detector(cfg, weights="runs/yolov8s_mask/weights/best.pt")
annotated_bgr, detections = det.detect_file("photo.jpg")
det.process_video("in.mp4", "out.mp4", every_n=2)
```

---

## Export and deployment

`scripts/export_model.py` writes, next to `best.pt`:

| Format | File | Use |
|---|---|---|
| ONNX (opset 12, simplified) | `best.onnx` | onnxruntime on any platform, TensorRT, OpenVINO |
| TFLite (float32 + float16) | `best_saved_model/*.tflite` | Android / iOS / Raspberry Pi |
| TorchScript | `best.torchscript` | PyTorch-native serving (C++ / TorchServe) |

Each export is reloaded and run on 5 test images; boxes must match the PyTorch model with
IoU > 0.99 and identical counts, otherwise the script exits non-zero and writes the details to
`export_report.json`. TFLite needs TensorFlow (installed automatically by Ultralytics, ~3 min)
and is attempted last so it can never block ONNX / TorchScript.

Run an exported model with the same API:

```python
from ultralytics import YOLO
YOLO("best.onnx", task="detect").predict("photo.jpg", imgsz=640, conf=0.45)
```

---

## Fallback: SSD-MobileNetV2

If the Ultralytics install is broken in your environment, `src/model.py` provides an SSD head on
a MobileNetV2 backbone assembled from `torchvision` blocks, with a plain PyTorch training loop in
`src/train.py::train_ssd` (AdamW, warm-up + cosine, early stopping, `last.pt` resume) and the same
`Detector(cfg, backend="ssd")` inference API. It is lighter and a little less accurate than YOLOv8-s.

---

## Colab robustness checklist

- Drive mounted first; dataset, checkpoints and figures all live there.
- `train_yolo` auto-resumes from `last.pt`; finished runs are detected and not restarted.
- `save_period=10` epoch checkpoints + `last.pt` every epoch.
- `free_memory()` (CUDA cache + GC) between training, evaluation and export.
- Keep-alive JavaScript cell in notebook 02.
- Every cell that can fail (download, credentials, GPU, file I/O, subprocess) raises a message
  saying what to do next.
- `seed=42` for Python, NumPy, PyTorch and CUDA; `deterministic=True` in Ultralytics.
- Pinned `requirements.txt` (including `fastapi`/`starlette` pins that Gradio 4.31 needs).

---

## License

MIT — see [LICENSE](LICENSE).

## Credits and citations

- **Dataset (Option A):** Larxel, *Face Mask Detection*, Kaggle, 2020.
  https://www.kaggle.com/datasets/andrewmvd/face-mask-detection (CC0). Original images from
  the MAFA and WIDER FACE datasets.
- **Roboflow Universe** face-mask projects (Option B) — see each project page for its license.
- **YOLOv8:** G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics (AGPL-3.0 for the library).
- **Albumentations:** A. Buslaev et al., *Albumentations: Fast and Flexible Image Augmentations*, Information 2020.
- **MAFA:** S. Ge et al., *Detecting Masked Faces in the Wild with LLE-CNNs*, CVPR 2017.
- **WIDER FACE:** S. Yang et al., *WIDER FACE: A Face Detection Benchmark*, CVPR 2016.
