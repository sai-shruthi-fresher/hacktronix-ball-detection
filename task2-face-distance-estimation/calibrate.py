
"""
Focal length calibration.

Stand at a KNOWN distance from the camera (measure with a ruler/tape),
look straight at the camera, and press SPACE to capture. This computes:

    f = (w_px * KNOWN_DISTANCE_M) / REAL_FACE_WIDTH_M

Result is saved to config.json for use by main.py
"""

import cv2
import json
import sys
from face_distance.detector import FaceDetector

KNOWN_DISTANCE_M = 0.60       # <-- measure this distance before running!
REAL_FACE_WIDTH_M = 0.15      # <-- average adult face width (~0.14-0.16m)


def main():
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        sys.exit(1)

    print(f"Stand exactly {KNOWN_DISTANCE_M} m from the camera.")
    print("Press SPACE to capture, ESC to quit.")

    focal_length = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        face = detector.detect(frame)
        if face:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"w_px={w}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Calibration - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32 and face:  # SPACE
            focal_length = (face["w"] * KNOWN_DISTANCE_M) / REAL_FACE_WIDTH_M
            print(f"Captured face width: {face['w']} px")
            print(f"Computed focal length: {focal_length:.2f} px")
            break

    cap.release()
    cv2.destroyAllWindows()

    if focal_length:
        config = {
            "focal_length_px": focal_length,
            "real_face_width_m": REAL_FACE_WIDTH_M,
            "known_distance_m": KNOWN_DISTANCE_M,
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Saved focal length to config.json")
    else:
        print("No capture made. Run again.")


if __name__ == "__main__":
    main()