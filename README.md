# MSMF: Spatial-Guided Multi-level Fusion for YOLOv12

This repository provides the **core implementation** of the proposed
**MSMF module (SGM + FMS + Fusion)** for multi-level feature fusion in YOLOv12,
aimed at robust small-object detection in complex ecological environments.

---

## Overview

MSMF is a lightweight multi-level feature fusion module that integrates
spatial guidance and global feature modulation to enhance small-object
representation under strong background interference.
The overall computational complexity is **O(C·H·W)**.

---

## Repository Structure

papercode/
├── MSMF.py # Proposed MSMF module
├── CoordAtt.py # Coordinate Attention module
├── DySample.py # Dynamic sampling module
├── yolov12-CA-DySample-MSMF.yaml # YOLOv12 model configuration
├── make_isood_small_split.py # Small-object subset construction script
└── heatmap.py # Visualization utility

yaml
复制代码

---

Dataset

Experiments are conducted on two datasets:

1. iSOOD (Public)

The publicly available iSOOD dataset is used for benchmarking.
It can be downloaded from:

https://zenodo.org/records/10903574

This dataset is not included in this repository.

2. FSOOD (Private)

We also evaluate our method on an additional FSOOD dataset, which is a private dataset collected under specific conditions.

Due to privacy and data protection policies, this dataset cannot be publicly released.
However, experimental results on FSOOD are reported in the paper for completeness.

Researchers interested in this dataset may contact the authors for further discussion.

---



