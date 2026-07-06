"""Evaluate YOLO-style detectors used in the CDM-YOLOv12 experiments."""

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
    parser = argparse.ArgumentParser(description="Evaluate a trained detector.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="Evaluation device, e.g. 0 or cpu.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default=None, help="Evaluation run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    val_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        project=args.project,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device
    if args.name:
        val_kwargs["name"] = args.name

    model.val(**val_kwargs)


if __name__ == "__main__":
    main()
