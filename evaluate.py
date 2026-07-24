
import os
import cv2
from ultralytics import YOLO

IMAGES_DIR = "data/images"
LABELS_DIR = "data/labels"
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.35
BALL_CLASS_ID = 32  # change to 0 if using a custom-trained single-class model

def yolo_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]

def iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def load_ground_truth(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            box = list(map(float, parts[1:5]))
            boxes.append(yolo_to_xyxy(box, img_w, img_h))
    return boxes

def main():
    model = YOLO("yolov8n.pt")

    if not os.path.exists(IMAGES_DIR):
        print(f"No images found in {IMAGES_DIR}. Add test images and YOLO-format")
        print(f"label files in {LABELS_DIR} to compute F1 score.")
        return

    tp, fp, fn = 0, 0, 0
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not image_files:
        print(f"No images found in {IMAGES_DIR}. Add test images and YOLO-format")
        print(f"label files in {LABELS_DIR} to compute F1 score.")
        return

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        gt_boxes = load_ground_truth(label_path, w, h)
        matched_gt = [False] * len(gt_boxes)

        results = model.predict(img, classes=[BALL_CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                pred_boxes.append(list(map(float, box.xyxy[0])))

        for pred in pred_boxes:
            best_iou, best_idx = 0, -1
            for i, gt in enumerate(gt_boxes):
                if matched_gt[i]:
                    continue
                score = iou(pred, gt)
                if score > best_iou:
                    best_iou, best_idx = score, i
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_gt[best_idx] = True
            else:
                fp += 1

        fn += matched_gt.count(False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Images evaluated: {len(image_files)}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

if __name__ == "__main__":
    main()