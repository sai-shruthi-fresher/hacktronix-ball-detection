
"""
Live monocular face distance & angle estimation.
Run calibrate.py once first to generate config.json.
"""

import cv2
import json
import os
import sys
from face_distance.detector import FaceDetector
from face_distance.estimator import DistanceEstimator

CONFIG_PATH = "config.json"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print("config.json not found. Run calibrate.py first.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    config = load_config()
    detector = FaceDetector()
    estimator = DistanceEstimator(
        focal_length_px=config["focal_length_px"],
        real_face_width_m=config["real_face_width_m"],
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        sys.exit(1)

    print("Press ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h_img, w_img = frame.shape[:2]
        image_center_x = w_img / 2.0

        face = detector.detect(frame)
        if face:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            depth_m, angle_deg = estimator.estimate(
                face_center_x_px=face["cx"],
                face_width_px=w,
                image_center_x_px=image_center_x,
            )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Depth: {depth_m:.2f} m", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle_deg:.1f} deg", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.line(frame, (int(image_center_x), 0), (int(image_center_x), h_img),
                  (255, 0, 0), 1)

        cv2.imshow("Monocular Face Distance Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()