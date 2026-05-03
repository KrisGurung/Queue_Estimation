## Test script to check accuracy of different YOLO posing models (n/m/l/x) to check accuracy of human object 
## detection and tracker accuracy. 

import cv2
import os
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH  = "airport.qt"
OUTPUT_DIR  = "runs/detect/track_boxes"
OUTPUT_FILE = "airport_boxes.avi"
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO("yolo11m-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"

w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

print(f"Tracking... output → {output_path}")

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, classes=[0], verbose=False, imgsz=1088)

    annotated = frame.copy()

    boxes = results[0].boxes
    if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes, boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            obj_id = int(track_id.item())

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID label
            cv2.putText(annotated, f"ID {obj_id}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(annotated)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done. {frame_count} frames written to {output_path}")
