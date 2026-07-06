"""Train YOLO-style detectors used in the CDM-YOLOv12 experiments.

This script is a thin reproducibility wrapper around the Ultralytics API.
It keeps the main training settings explicit and allows the model and
dataset YAML files to be changed from the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPO_ROOT / "yolov12" / "yolov12-main"
if YOLO_ROOT.exists():
    sys.path.insert(0, str(YOLO_ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO-style detector.")
    parser.add_argument("--model", required=True, help="Path to model YAML or weights.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=None, help="Experiment name.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        patience=args.patience,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=False,
        seed=0,
        deterministic=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.name:
        train_kwargs["name"] = args.name

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
