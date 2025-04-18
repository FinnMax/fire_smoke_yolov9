# YOLOv9 Training and Inference for Fire and Smoke Detection
This section describes how to use YOLOv9 for training and inference to detect fire and smoke classes using fire and smoke datasets. The repository includes two Jupyter notebooks: `train.ipynb` for preprocessing data and training the model, and `inference.ipynb` for running validation and detection on new images or videos.

## Overview
The notebooks process the fire and smoke datasets in COCO format, convert annotations to YOLO format (mapping `smoke` to class 0 and `fire` to class 1), train a YOLOv9 model (`gelan-c` architecture), and perform inference to detect fire and smoke in images or videos. The training notebook includes data preprocessing and model training, while the inference notebook validates the trained model and runs detection.

Requirements
Environment: A Linux-based system (e.g., Ubuntu 20.04 or later) or Windows with GPU support recommended.
Python: Version 3.8 or later (compatible with YOLOv9 dependencies).
Dependencies: Install via `pip`:

```pip install torch torchvision tqdm ultralytics```
