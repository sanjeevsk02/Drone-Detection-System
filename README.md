# Real-Time Drone Detection and Counting System Using YOLOv8

A complete deep learning pipeline for detecting, tracking, and counting drones in images, videos, and live camera feeds. Built with YOLOv8n fine-tuned on the [UAV/Drone Dataset](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav) by Mehdi Özel.

## Results

| Metric | Value |
|--------|-------|
| mAP50 (test set) | 98.43% |
| mAP50-95 | 68.01% |
| Precision | 98.29% |
| Recall | 98.29% |
| Inference time | 13.9 ms (~72 fps) |

## Features

- **Image detection** — upload an image and get instant bounding box results with confidence scores
- **Video detection** — frame-by-frame processing with ByteTrack multi-drone tracking and persistent color-coded IDs
- **Live camera** — real-time WebRTC webcam feed with FPS display and running drone count
- **Virtual line crossing counter** — place a horizontal or vertical line anywhere in the frame to count how many drones cross it
- **Evaluation tab** — run or browse formal test-set evaluations with per-image CSV results

## Project Structure

```
drone-detection-system/
├── app.py                        # Streamlit web application
├── src/
│   ├── detect.py                 # DroneDetector class (inference + ByteTrack + drawing)
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Formal test-set evaluation
│   └── download_kaggle.py        # Dataset download and XML→YOLO conversion
├── configs/
│   ├── bytetrack_drone.yaml      # ByteTrack tracker configuration
│   └── drone_dataset.yaml        # YOLO dataset config
├── detection_limit_test.py       # Scale ablation study script
├── requirements.txt
└── README.md
```

`data/` (dataset) and `runs/` (trained weights and results) are excluded from the repository via `.gitignore`.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav) and place all images and XML annotation files into a single folder, then run:

```bash
python src/download_kaggle.py --local --src path/to/raw/files
```

This converts Pascal VOC XML annotations to YOLO format and splits the data into train (80%), validation (10%), and test (10%) sets.

## Training

```bash
python src/train.py
```

Trains YOLOv8n for 100 epochs on the drone dataset. Best weights are saved to `runs/train/drone_detector/weights/best.pt`. Per-epoch metrics are printed during training.

## Running the App

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal. The sidebar has a confidence threshold slider. The app requires a trained model — run training first.

### Image Tab
Upload a JPG/PNG image. The app shows the original and annotated result side by side with detection metrics and a download button.

### Video Tab
Upload an MP4/AVI/MOV file and click **Run Detection**. Enable object tracking to assign persistent IDs to each drone. Use the line crossing counter expander to place a virtual line and count drone crossings.

### Live Camera Tab
Click **Start** to open your webcam. The model runs in real time showing drones detected per frame, total unique drones seen, and live FPS.

### Evaluation Tab
Click **Run Evaluation on Test Set** to evaluate the model on the held-out 111 test images. Results include precision, recall, mAP50, mAP50-95, per-image CSV, and sample visualizations.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n (nano) |
| Parameters | 3.2M |
| Input size | 640×640 |
| Classes | 1 (drone) |
| Pretrained on | COCO |
| Tracker | ByteTrack |

## Detection Limit

A scale-based ablation study on all 111 test images shows the model maintains above 93% detection rate when images are shrunk to 10% of original size. Detection degrades meaningfully below 7% scale. The practical minimum detectable drone size is approximately 5–10 pixels in a 640×640 frame.

## Dataset Credit

Dataset by Mehdi Özel — [kaggle.com/datasets/dasmehdixtr/drone-dataset-uav](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav)
