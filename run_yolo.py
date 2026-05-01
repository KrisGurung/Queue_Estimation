import cv2
import os
import json
from ultralytics import solutions

# 1. Open the video source
video_path = "video.qt"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 2. Define measurement line across the middle of the screen
line_pts = [(0, h // 2), (w, h // 2)]

# 3. Initialize the SpeedEstimator with the model
speed_obj = solutions.SpeedEstimator(
    region=line_pts,
    model="yolo11n.pt",  # Automatically uses this model for tracking
    show=False,
)

# 4. Ensure output directory exists and setup VideoWriter
output_dir = "runs/detect/track"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "speed_estimation.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

print(f"Starting tracking & speed estimation... Output will be saved to {output_path}")

# 5. Process the video frame-by-frame
tracking_data = []
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Process frame naturally processes tracking and calculates speed automatically
    results = speed_obj.process(frame)
    annotated_frame = results.plot_im
    
    # Custom Annotation: Extract 2D Center Coordinate of each object
    # speed_obj.tracks contains the YOLO Results object for the current frame
    if speed_obj.tracks:
        boxes = speed_obj.tracks[0].boxes if isinstance(speed_obj.tracks, list) else speed_obj.tracks.boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):
                # The xywh format returns [x_center, y_center, width, height]
                # We pull the center coordinate (respecting the camera plane)
                x_center, y_center, _, _ = box.xywh[0].cpu().numpy()
                cx, cy = int(x_center), int(y_center)
                obj_id = int(track_id.item())
                # Extract speed if calculated
                obj_speed = results.speed_dict.get(obj_id, 0.0) if hasattr(results, 'speed_dict') else 0.0
                
                # Log the data for this frame
                tracking_data.append({
                    "frame": frame_count,
                    "object_id": obj_id,
                    "x": cx,
                    "y": cy,
                    "speed": obj_speed
                })
                
                # Draw the (X, Y) text near the bottom center of the detected object
                text = f"({cx},{cy})"
                cv2.putText(annotated_frame, text, (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
    
    # Write the frame out
    out.write(annotated_frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing video complete! Exporting JSON...")
json_path = os.path.join(output_dir, "positional_data.json")
with open(json_path, "w") as json_file:
    json.dump(tracking_data, json_file, indent=4)
print(f"Tracking data saved to {json_path}")
