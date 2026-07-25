
# Monocular Face Distance Estimation

HackTronix 2.0 — Track B Qualifier, Task 2

Estimates face depth (Z) and horizontal deviation angle (θ) from a single
2D webcam image using the pinhole camera model.

## Method

- **Detection**: OpenCV Haar Cascade frontal face detector
- **Depth**: `Z = (f × W) / w_px`
- **Angle**: `θ = arctan((x − c_x) / f)`

Where:
| Symbol | Meaning |
|---|---|
| f | Focal length (pixels), obtained via calibration |
| W | Real average face width (meters), ~0.15 |
| w_px | Detected face width in pixels |
| x | Face center x-coordinate (pixels) |
| c_x | Image center x-coordinate (pixels) |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Calibrate (once per camera)
Measure a known distance (e.g. 60 cm) from your webcam, then run:
```bash
python calibrate.py
```
Look at the camera, press **SPACE** when your face box appears steady.
This saves `config.json` with your camera's focal length.

### 2. Run live estimation
```bash
python main.py
```
Displays a live feed with depth (meters) and angle (degrees) overlaid.
Press **ESC** to quit.

## Accuracy Notes

- Accuracy depends heavily on calibration quality — recalibrate if you
  change cameras or resolution.
- Real face width varies per person (~0.14–0.16 m); using the population
  average introduces the expected ±50–150 cm-scale error tolerated by
  the task spec.
- Works best in good, even lighting with the face roughly facing the camera.

## Project Structure

```
face-distance-estimation/
├── face_distance/
│   ├── detector.py      # Face detection wrapper
│   └── estimator.py     # Depth/angle math
├── calibrate.py          # One-time focal length calibration
├── main.py                # Live demo
└── requirements.txt
```