# Pre-trained Weights

This directory contains the pre-trained weights used for the main iSOOD experiments.

| File | Description | Size | SHA256 |
| --- | --- | ---: | --- |
| `YOLOv12n_baseline_iSOOD_best.pt` | YOLOv12n baseline trained on the iSOOD small-object subset | 5,533,099 bytes | `AFD1EEEBE7D1F9C0A0127890F0C21986E09AF6AAF17FB1D39EC76D2E5A9BA09A` |
| `CDM-YOLOv12_iSOOD_best.pt` | Proposed CDM-YOLOv12 model trained on the iSOOD small-object subset | 5,379,930 bytes | `1386613FB805109DCE93726AF4C5055BB10A9634951DA45B712E645FE9DD97D7` |

Example evaluation command:

```bash
python scripts/eval_yolo.py \
  --weights weights/CDM-YOLOv12_iSOOD_best.pt \
  --data configs/datasets/isood_small.yaml \
  --imgsz 640 --batch 32 --split test \
  --name cdmyolov12_isood_test
```
