"""Masked face detection package.

Modules
-------
utils      : config loading, seeding, device selection, drawing helpers
dataset    : VOC->YOLO conversion, stratified splitting, Albumentations dataset
model      : YOLOv8 wrapper + SSD-MobileNet fallback
train      : training entry point with auto-resume
evaluate   : metrics, confusion matrix, PR curves, failure analysis, FPS
inference  : image / video / frame detection with annotated output
"""

__version__ = "1.0.0"
