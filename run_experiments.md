# Reproducing the Main Experiments

This document summarizes the commands used to train and evaluate the YOLO-style detectors in the CDM-YOLOv12 experiments. Run the commands from the repository root.

## Environment

Install the local YOLOv12/Ultralytics-style implementation before running the experiments:

```bash
cd yolov12/yolov12-main
pip install -r requirements.txt
pip install -e .
cd ../..
```

## Dataset YAML Files

Prepare the dataset YAML files before training:

```text
configs/datasets/isood_small.yaml
configs/datasets/fsood.yaml
```

Each dataset YAML should follow the Ultralytics detection format:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: sewage_outfall
```

The iSOOD dataset is publicly available. The FSOOD dataset is private; access instructions should be provided separately in `DATASET_ACCESS.md`.

## Train CDM-YOLOv12

```bash
python scripts/train_yolo.py \
  --model yolov12-CA-DySample-MSMF.yaml \
  --data configs/datasets/isood_small.yaml \
  --epochs 300 --imgsz 640 --batch 32 \
  --name cdmyolov12_isood
```

```bash
python scripts/train_yolo.py \
  --model yolov12-CA-DySample-MSMF.yaml \
  --data configs/datasets/fsood.yaml \
  --epochs 300 --imgsz 640 --batch 32 \
  --name cdmyolov12_fsood
```

## Evaluate CDM-YOLOv12

```bash
python scripts/eval_yolo.py \
  --weights runs/detect/cdmyolov12_isood/weights/best.pt \
  --data configs/datasets/isood_small.yaml \
  --imgsz 640 --batch 32 --split test \
  --name cdmyolov12_isood_test
```

```bash
python scripts/eval_yolo.py \
  --weights runs/detect/cdmyolov12_fsood/weights/best.pt \
  --data configs/datasets/fsood.yaml \
  --imgsz 640 --batch 32 --split test \
  --name cdmyolov12_fsood_test
```

## Train YOLO Baselines

The same training wrapper can be used for YOLOv5n, YOLOv8n, YOLOv10n, YOLOv11n, and YOLOv12n by changing the model YAML:

```bash
python scripts/train_yolo.py --model configs/baselines/yolov5n.yaml  --data configs/datasets/isood_small.yaml --epochs 300 --imgsz 640 --batch 32 --name yolov5n_isood
python scripts/train_yolo.py --model configs/baselines/yolov8n.yaml  --data configs/datasets/isood_small.yaml --epochs 300 --imgsz 640 --batch 32 --name yolov8n_isood
python scripts/train_yolo.py --model configs/baselines/yolov10n.yaml --data configs/datasets/isood_small.yaml --epochs 300 --imgsz 640 --batch 32 --name yolov10n_isood
python scripts/train_yolo.py --model configs/baselines/yolov11n.yaml --data configs/datasets/isood_small.yaml --epochs 300 --imgsz 640 --batch 32 --name yolov11n_isood
python scripts/train_yolo.py --model configs/baselines/yolov12n.yaml --data configs/datasets/isood_small.yaml --epochs 300 --imgsz 640 --batch 32 --name yolov12n_isood
```

The same commands can be applied to FSOOD by replacing `configs/datasets/isood_small.yaml` with `configs/datasets/fsood.yaml` and updating the run names.

## Evaluate YOLO Baselines

```bash
python scripts/eval_yolo.py --weights runs/detect/yolov5n_isood/weights/best.pt  --data configs/datasets/isood_small.yaml --imgsz 640 --batch 32 --split test --name yolov5n_isood_test
python scripts/eval_yolo.py --weights runs/detect/yolov8n_isood/weights/best.pt  --data configs/datasets/isood_small.yaml --imgsz 640 --batch 32 --split test --name yolov8n_isood_test
python scripts/eval_yolo.py --weights runs/detect/yolov10n_isood/weights/best.pt --data configs/datasets/isood_small.yaml --imgsz 640 --batch 32 --split test --name yolov10n_isood_test
python scripts/eval_yolo.py --weights runs/detect/yolov11n_isood/weights/best.pt --data configs/datasets/isood_small.yaml --imgsz 640 --batch 32 --split test --name yolov11n_isood_test
python scripts/eval_yolo.py --weights runs/detect/yolov12n_isood/weights/best.pt --data configs/datasets/isood_small.yaml --imgsz 640 --batch 32 --split test --name yolov12n_isood_test
```

## Non-YOLO Baselines

Faster R-CNN and RT-DETR follow the official default training recipes of their corresponding implementations. Their baseline configuration summaries are provided in:

```text
configs/baselines/faster_rcnn.yaml
configs/baselines/rt_detr.yaml
```

If these baselines are re-run, keep the same train/validation/test splits, input resolution, hardware platform, evaluation metrics, and best-validation-checkpoint selection criterion as the YOLO-style detectors.
