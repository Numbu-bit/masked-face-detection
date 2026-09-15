"""Training entry point.

``train_yolo``  : maps ``configs/default.yaml`` onto Ultralytics' ``model.train``
                  and auto-resumes from ``last.pt`` if the Colab runtime died.
``train_ssd``   : plain PyTorch loop for the SSD-MobileNetV2 fallback with
                  AdamW + cosine schedule, tqdm progress and periodic checkpoints.

CLI::

    python -m src.train --data /content/data/data.yaml [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import (
    ensure_dir,
    find_last_checkpoint,
    free_memory,
    get_device,
    get_logger,
    load_config,
    resolve_save_dir,
    run_dir,
    save_config,
    set_seed,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# YOLOv8
# --------------------------------------------------------------------------- #


def yolo_train_kwargs(cfg: Dict[str, Any], data_yaml: str, device: str) -> Dict[str, Any]:
    """Translate our config into Ultralytics ``train()`` keyword arguments.

    Keeping this in one function means notebook 02 and the CLI cannot drift.

    Args:
        cfg: Loaded configuration.
        data_yaml: Path to ``data.yaml``.
        device: ``"0"`` or ``"cpu"``.

    Returns:
        Keyword arguments for ``ultralytics.YOLO.train``.
    """
    return dict(
        data=str(data_yaml),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["image_size"]),
        batch=int(cfg["batch_size"]),
        workers=int(cfg["num_workers"]),
        patience=int(cfg["patience"]),
        device=device,
        project=str(resolve_save_dir(cfg)),
        name=str(cfg["run_name"]),
        exist_ok=True,  # re-running the cell must not create run_name2, run_name3, ...
        save=True,
        save_period=int(cfg["save_period"]),
        plots=True,
        verbose=True,
        seed=int(cfg["seed"]),
        deterministic=True,
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        cos_lr=(str(cfg["scheduler"]).lower() == "cosine"),
        warmup_epochs=float(cfg["warmup_epochs"]),
        augment=bool(cfg["augment"]),
        mosaic=float(cfg["mosaic"]),
        mixup=float(cfg["mixup"]),
        fliplr=float(cfg["albumentations"]["horizontal_flip_p"]),
        scale=float(cfg["albumentations"]["random_scale_limit"]),
    )


def train_yolo(
    cfg: Dict[str, Any],
    data_yaml: str,
    resume: Optional[bool] = None,
    epochs_override: Optional[int] = None,
) -> Path:
    """Fine-tune YOLOv8 with auto-resume.

    Behaviour:
    * If ``resume`` is ``None`` the function looks for ``last.pt`` in the run
      directory and resumes when found; otherwise it starts fresh.
    * A resumed run continues with the *original* arguments Ultralytics
      stored in the checkpoint, exactly as the Ultralytics docs require.

    Args:
        cfg: Loaded configuration.
        data_yaml: Path to ``data.yaml`` produced by notebook 01.
        resume: Force resume (True) / fresh start (False) / auto (None).
        epochs_override: Handy for smoke tests (``epochs=1``).

    Returns:
        Path to ``best.pt``.

    Raises:
        FileNotFoundError: If ``data_yaml`` does not exist.
        RuntimeError: If training finishes without producing ``best.pt``.
    """
    from src.model import build_yolo

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"{data_yaml} not found - run notebook 01 (data preparation) first.")

    set_seed(int(cfg["seed"]))
    device = get_device().device
    out_dir = ensure_dir(run_dir(cfg))
    save_config(cfg, out_dir / "config_snapshot.yaml")

    last = find_last_checkpoint(cfg)
    do_resume = (last is not None) if resume is None else bool(resume)
    if do_resume and last is None:
        log.warning("resume requested but no last.pt found; starting a fresh run.")
        do_resume = False

    t0 = time.time()
    if do_resume:
        log.info("Resuming from %s", last)
        model = build_yolo(cfg, weights=str(last))
        try:
            model.train(resume=True)
        except AssertionError as exc:
            # Ultralytics asserts when last.pt already reached its final epoch.
            if "nothing to resume" in str(exc).lower() or "is finished" in str(exc).lower():
                log.info("Previous run already completed all epochs - nothing to resume. "
                         "Delete %s or change run_name to train again.", out_dir)
            else:
                raise
    else:
        model = build_yolo(cfg)
        kwargs = yolo_train_kwargs(cfg, data_yaml, device)
        if epochs_override is not None:
            kwargs["epochs"] = int(epochs_override)
        log.info("Starting training: %s", {k: kwargs[k] for k in ("epochs", "imgsz", "batch", "device")})
        model.train(**kwargs)
    log.info("Training finished in %.1f min", (time.time() - t0) / 60)

    best = out_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training ended but {best} is missing. Check the Ultralytics log above.")
    free_memory()
    return best


# --------------------------------------------------------------------------- #
# SSD-MobileNetV2 fallback
# --------------------------------------------------------------------------- #


def train_ssd(cfg: Dict[str, Any], data_root: str, epochs_override: Optional[int] = None) -> Path:
    """Train the fallback SSD model with a hand-written PyTorch loop.

    Uses AdamW + warmup/cosine LR, saves ``last.pt`` every epoch and
    ``best.pt`` on the lowest validation loss, and resumes automatically.

    Args:
        cfg: Loaded configuration.
        data_root: Folder holding ``train/ val/ test/`` (from notebook 01).
        epochs_override: Handy for smoke tests.

    Returns:
        Path to ``best.pt``.
    """
    import torch
    from tqdm.auto import tqdm

    from src.dataset import make_dataloader
    from src.model import build_ssd_mobilenetv2

    set_seed(int(cfg["seed"]))
    dev_info = get_device()
    device = torch.device("cuda:0" if dev_info.is_gpu else "cpu")
    out_dir = ensure_dir(run_dir(cfg) / "weights")
    epochs = int(epochs_override or cfg["epochs"])

    train_loader = make_dataloader(Path(data_root), "train", cfg, augment=True)
    val_loader = make_dataloader(Path(data_root), "val", cfg, augment=False)

    model = build_ssd_mobilenetv2(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    warmup = int(cfg["warmup_epochs"])

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, best_val = 0, float("inf")
    last_ckpt = out_dir / "last.pt"
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        log.info("Resumed SSD training from epoch %d", start_epoch)

    patience, bad_epochs = int(cfg["patience"]), 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for images, targets in bar:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            running += float(loss)
            bar.set_postfix(loss=f"{float(loss):.3f}")
        scheduler.step()
        train_loss = running / max(1, len(train_loader))

        val_loss = _ssd_val_loss(model, val_loader, device)
        log.info("epoch %d | train %.4f | val %.4f | lr %.2e", epoch + 1, train_loss, val_loss, optimizer.param_groups[0]["lr"])

        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val": min(best_val, val_loss),
        }
        torch.save(state, last_ckpt)
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save(state, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping after %d epochs without improvement", patience)
                break
        if (epoch + 1) % int(cfg["save_period"]) == 0:
            torch.save(state, out_dir / f"epoch{epoch + 1}.pt")

    free_memory()
    return out_dir / "best.pt"


def _ssd_val_loss(model: Any, loader: Any, device: Any) -> float:
    """Mean SSD loss over a loader (torchvision only returns losses in train mode)."""
    import torch

    model.train()  # loss computation requires train mode; no grads are taken
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            total += float(sum(model(images, targets).values()))
    model.eval()
    return total / max(1, len(loader))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the masked-face detector")
    parser.add_argument("--data", required=True, help="Path to data.yaml (YOLO) or data root (SSD)")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/default.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke tests)")
    parser.add_argument("--backend", choices=["yolo", "ssd"], default="yolo")
    parser.add_argument("--no-resume", action="store_true", help="Ignore last.pt and start fresh")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend == "yolo":
        best = train_yolo(cfg, args.data, resume=False if args.no_resume else None, epochs_override=args.epochs)
    else:
        best = train_ssd(cfg, args.data, epochs_override=args.epochs)
    log.info("Best weights: %s", best)


if __name__ == "__main__":
    main()
