
# HackTronix 2.0 - Track B, Task 1: Ball Detection

Real-time ball detection system using YOLOv8, optimized for maximum F1 score and maximum FPS on a 2D camera feed.

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Usage

Run real-time detection on webcam:

python src/detect_webcam.py

Press q to quit. FPS is shown live in the top-left corner.

Run detection on a static test image:

python src/detect_image.py

## Approach

- Model: YOLOv8n (nano) - best FPS/accuracy trade-off on CPU.
- Baseline: pretrained COCO weights, filtered to the "sports ball" class (class ID 32).
- Confidence threshold: 0.35