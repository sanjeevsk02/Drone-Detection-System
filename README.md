# Drone Detection System

A drone detection system built with YOLOv8, trained on the [UAV/Drone Dataset](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav) by Mehdi Özel.

## Results

| Metric | Score |
|--------|-------|
| mAP50 | ~96% |
| mAP50-95 | ~68% |
| Precision | ~95% |
| Recall | ~91% |

## Project Structure

```
drone detection system/
├── app.py                  # Streamlit web UI
├── src/
│   ├── train.py            # Model training
│   ├── detect.py           # Detection inference
│   └── download_kaggle.py  # Dataset download & XML→YOLO conversion
├── configs/
│   └── drone_dataset.yaml  # Dataset config
├── data/                   # Dataset (not included)
└── runs/                   # Training outputs (not included)
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the dataset from Kaggle and place all images and XML files into a single folder, then run:

```bash
python src/download_kaggle.py --local --src path/to/raw/files
```

This converts Pascal VOC XML annotations to YOLO format and splits into train/val/test.

## Training

```bash
python src/train.py
```

Per-epoch metrics (mAP50, Precision, Recall) are printed during training. Best weights are saved to `runs/train/drone_detector/weights/best.pt`.

## Detection (Web UI)

```bash
streamlit run app.py
```

Upload an image or video — the app will draw bounding boxes and show a **DRONE DETECTED** alert if a drone is found.

## Model

- Architecture: YOLOv8n (nano)
- Input size: 640×640
- Classes: 1 (drone)
- Pretrained on COCO, fine-tuned on UAV dataset

## Dataset Credit

Dataset by Mehdi Özel — [kaggle.com/datasets/dasmehdixtr/drone-dataset-uav](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav)
