# CDM-YOLOv12 for Small Sewage Outfall Detection

This repository provides the implementation and reproducibility materials for CDM-YOLOv12, a lightweight modulation fusion detector for small sewage outfall detection in complex ecological scenes. The method is built on YOLOv12n and integrates Coordinate Attention, DySample, and the proposed Multi-Scale Modulation Fusion (MSMF) module.

## Main Components

- `MSMF.py`: implementation of the proposed MSMF module.
- `CoordAtt.py`: Coordinate Attention module.
- `DySample.py`: content-adaptive upsampling module.
- `yolov12-CA-DySample-MSMF.yaml`: CDM-YOLOv12 model configuration.
- `make_isood_small_split.py`: script for constructing the iSOOD small-object subset.
- `heatmap.py`: visualization utility for activation analysis.
- `scripts/train_yolo.py`: training wrapper for YOLO-style detectors.
- `scripts/eval_yolo.py`: evaluation wrapper for trained YOLO-style detectors.
- `run_experiments.md`: command examples for reproducing the main experiments.

The local YOLOv12/Ultralytics-style codebase is included under:

```text
yolov12/yolov12-main/
```

## Repository Structure

```text
papercode/
|-- CoordAtt.py
|-- DySample.py
|-- MSMF.py
|-- heatmap.py
|-- make_isood_small_split.py
|-- yolov12-CA-DySample-MSMF.yaml
|-- DATASET_ACCESS.md
|-- run_experiments.md
|-- configs/
|   |-- baselines/
|   |   |-- faster_rcnn.yaml
|   |   |-- rt_detr.yaml
|   |   |-- yolov5n.yaml
|   |   |-- yolov8n.yaml
|   |   |-- yolov10n.yaml
|   |   |-- yolov11n.yaml
|   |   `-- yolov12n.yaml
|   `-- datasets/
|       |-- fsood.yaml
|       `-- isood_small.yaml
|-- scripts/
|   |-- train_yolo.py
|   `-- eval_yolo.py
|-- weights/
|   |-- CDM-YOLOv12_iSOOD_best.pt
|   |-- YOLOv12n_baseline_iSOOD_best.pt
|   `-- README.md
`-- yolov12/
    `-- yolov12-main/
```

## Installation

Install the local YOLOv12/Ultralytics-style implementation:

```bash
cd yolov12/yolov12-main
pip install -r requirements.txt
pip install -e .
cd ../..
```

The experiments in the paper were conducted with PyTorch 2.1, CUDA 12.6, and Python 3.10.

## Dataset Preparation

The dataset configuration files are provided in:

```text
configs/datasets/isood_small.yaml
configs/datasets/fsood.yaml
```

Before training or evaluation, update the `path` field in each YAML file to match the local dataset directory:

```yaml
path: /path/to/iSOOD_10481_v1.0_small_split
train: images/train
val: images/val
test: images/test

names:
  0: sewage_outfall
```

The iSOOD dataset is publicly available from:

```text
https://zenodo.org/records/10903574
```

The iSOOD small-object subset can be generated using:

```bash
python make_isood_small_split.py
```

FSOOD is not publicly released at this stage due to regional data management requirements, field-site sensitivity, and privacy-related restrictions. Access requests can be made following the instructions in `DATASET_ACCESS.md`.

## Baseline Configurations

Configuration files for the baseline models are provided in:

```text
configs/baselines/
```

The directory includes YOLOv5n, YOLOv8n, YOLOv10n, YOLOv11n, YOLOv12n, Faster R-CNN, and RT-DETR baseline configurations. The YOLO-style baselines can be trained using the same wrapper script by changing the `--model` argument. Faster R-CNN and RT-DETR follow the official default recipes of their corresponding implementations, with configuration summaries included for reproducibility.

## Pre-trained Weights

The following pre-trained weights are included for the main iSOOD experiments:

```text
weights/YOLOv12n_baseline_iSOOD_best.pt
weights/CDM-YOLOv12_iSOOD_best.pt
```

File descriptions and SHA256 checksums are provided in `weights/README.md`.

## Training

Train CDM-YOLOv12 on the iSOOD small-object subset:

```bash
python scripts/train_yolo.py \
  --model yolov12-CA-DySample-MSMF.yaml \
  --data configs/datasets/isood_small.yaml \
  --epochs 300 --imgsz 640 --batch 32 \
  --name cdmyolov12_isood
```

Train the YOLOv12n baseline on the same dataset:

```bash
python scripts/train_yolo.py \
  --model configs/baselines/yolov12n.yaml \
  --data configs/datasets/isood_small.yaml \
  --epochs 300 --imgsz 640 --batch 32 \
  --name yolov12n_isood
```

Additional commands for YOLOv5n, YOLOv8n, YOLOv10n, YOLOv11n, YOLOv12n, and CDM-YOLOv12 are listed in `run_experiments.md`.

## Evaluation

Evaluate the provided CDM-YOLOv12 iSOOD weight:

```bash
python scripts/eval_yolo.py \
  --weights weights/CDM-YOLOv12_iSOOD_best.pt \
  --data configs/datasets/isood_small.yaml \
  --imgsz 640 --batch 32 --split test \
  --name cdmyolov12_isood_test
```

Evaluate the provided YOLOv12n baseline weight:

```bash
python scripts/eval_yolo.py \
  --weights weights/YOLOv12n_baseline_iSOOD_best.pt \
  --data configs/datasets/isood_small.yaml \
  --imgsz 640 --batch 32 --split test \
  --name yolov12n_isood_test
```

## FSOOD Access

The FSOOD dataset was collected from real field inspection scenes. Because the images are associated with specific local environments and inspection sites, the dataset is not publicly released at this stage. Researchers interested in non-commercial academic use may request access by contacting:

```text
zhaoji_ustl@outlook.com
```

Please see `DATASET_ACCESS.md` for the required request information and usage restrictions.

## Notes

The scripts in this repository expose the main experimental settings used in the paper, including 300 training epochs, 640 x 640 input resolution, batch size 32, SGD optimizer, fixed seed, deterministic training, mosaic augmentation, and MixUp disabled. Users should keep the same data splits and evaluation protocol when reproducing the reported results.
