# YOLOv9 Training and Inference for Fire and Smoke Detection
This section describes how to use YOLOv9 for training and inference to detect fire and smoke classes using fire and smoke datasets. The repository includes two Jupyter notebooks: `train.ipynb` for preprocessing data and training the model, and `inference.ipynb` for running validation and detection on new images or videos.

## Overview
The notebooks process the fire and smoke datasets in COCO format, convert annotations to YOLO format (mapping `smoke` to class 0 and `fire` to class 1), train a YOLOv9 model (`gelan-c` architecture), and perform inference to detect fire and smoke in images or videos. The training notebook includes data preprocessing and model training, while the inference notebook validates the trained model and runs detection.

Requirements
Environment: A Linux-based system (e.g., Ubuntu 20.04 or later) or Windows with GPU support recommended.
Python: Version 3.8 or later (compatible with YOLOv9 dependencies).
Dependencies: Install via `pip`:

```bash
pip install torch torchvision tqdm ultralytics
```
YOLOv9 Repository: Clone the YOLOv9 repository and place the notebooks in its root directory:
```bash
git clone https://github.com/WongKinYiu/yolov9
```
Hardware: A CUDA-enabled GPU (e.g., NVIDIA) for faster training and inference (set `--device 0` in the code).
Dataset: Fire and smoke datasets in COCO format, with annotations (`instances_default.json`) and images split into `train` and `val` folders.

### Training with train.ipynb ###
The `train.ipynb` notebook performs the following:

Converts COCO annotations to YOLO format for fire (`class 1`) and smoke (`class 0)`.
Trains a YOLOv9 `gelan-c` model for 300 epochs with optimized hyperparameters.

This trains the model and saves weights to `/yolov9/runs/train/gelan_c_300ep_smoke_opt/weights/best.pt.`

### Notes ###
Update `your_main_directory` in the notebook to your project path.
Ensure `train.yaml` is configured with paths to `train` and `val` datasets and class names (`smoke`, `fire`).
Training requires significant GPU memory; adjust `--batch-size` if needed.

## Inference with inference.ipynb ##
The `inference.ipynb` notebook validates the trained model and runs detection on validation images or new data.

### Notes ###
Update paths (e.g., `/train.yaml`, `/dataset_val/images/val`) to match your setup.
Adjust `--conf` (confidence threshold) or `--iou` for stricter or looser detections.
The `--source` can be a folder, image, or video for flexible inference.

