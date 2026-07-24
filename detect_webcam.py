import cv2
import time
from ultralytics import YOLO

# COCO class id for "sports ball"
BALL_CLASS_ID = 32

def main():
    # yolov8n = "nano" = smallest/fastest model. Good starting point for max FPS.
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection, keep only the "sports ball" class, confidence >= 0.35
        results = model.predict(frame, classes=[BALL_CLASS_ID], conf=0.35, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ball {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Ball Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    