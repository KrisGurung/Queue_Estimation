import cv2
import os
import json
import math
from ultralytics import YOLO

# ── COCO keypoint indices ────────────────────────────────────────────────────
LEFT_SHOULDER,  RIGHT_SHOULDER  = 5,  6
LEFT_HIP,       RIGHT_HIP       = 11, 12
LEFT_KNEE,      RIGHT_KNEE      = 13, 14
LEFT_ANKLE,     RIGHT_ANKLE     = 15, 16


def is_standing(keypoints, conf_thresh=0.3, leg_torso_ratio=0.8):
    """
    Return True if a person's pose looks like they are standing/walking.

    Logic (perspective-agnostic):
      - Compute torso height  = hip_y  - shoulder_y   (pixels, downward +)
      - Compute leg extension = ankle_y - hip_y
      - ratio = leg_extension / torso_height
      - Standing people have fully extended legs → ratio ≥ leg_torso_ratio
      - Sitting people have bent/raised legs    → ratio < leg_torso_ratio

    Falls back to knees when ankles are not visible (e.g. partial occlusion).
    Returns True (standing assumed) when keypoint confidence is too low.
    """
    kp = keypoints.cpu().numpy()  # shape [17, 3]  (x, y, confidence)

    def get_y(idx):
        """Return y-coord if keypoint confidence meets threshold, else None."""
        _, y, conf = kp[idx]
        return float(y) if conf >= conf_thresh else None

    shoulder_ys = [v for v in [get_y(LEFT_SHOULDER), get_y(RIGHT_SHOULDER)] if v is not None]
    hip_ys      = [v for v in [get_y(LEFT_HIP),      get_y(RIGHT_HIP)]      if v is not None]
    knee_ys     = [v for v in [get_y(LEFT_KNEE),     get_y(RIGHT_KNEE)]     if v is not None]
    ankle_ys    = [v for v in [get_y(LEFT_ANKLE),    get_y(RIGHT_ANKLE)]    if v is not None]

    # Need at least shoulders + hips to compute torso reference
    if not shoulder_ys or not hip_ys:
        return False  # Not enough info → keep the detection

    #KFIX1: No knees and ankle detection mean no person detected
    if not ankle_ys or not knee_ys:
        return False

    shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
    hip_y      = sum(hip_ys)      / len(hip_ys)
    torso_h    = hip_y - shoulder_y

    if torso_h <= 0:
        return False  # Upside-down / abnormal → keep

    if ankle_ys:
        leg_ref_y = sum(ankle_ys) / len(ankle_ys)
        threshold = leg_torso_ratio

    elif knee_ys:
        # Knees are roughly halfway down the leg, so halve the threshold
        leg_ref_y = sum(knee_ys) / len(knee_ys)
        threshold = leg_torso_ratio * 0.5
    else:
        return False  # No lower-body keypoints → keep

    ratio = (leg_ref_y - hip_y) / torso_h
    return ratio >= threshold


# ── Counter / exclusion zone ─────────────────────────────────────────────────
# Define the pixel rectangle that covers the counter (server side).
# Any person whose bounding-box centre falls inside this region is excluded
# from queue clustering entirely.
#
# Format: (x1, y1, x2, y2)  – top-left and bottom-right corners in pixels.
# Set to None to disable the exclusion zone.
#
# TIP: run the script once and read off the (cx, cy) labels printed on the
#      server's bounding box to determine the right coordinates.
#
# KEDIT3: Counter exclusion zone  ← adjust these coordinates for your video
COUNTER_ZONE = (320, 0, 640, 250)   # example: left strip covering the counter


def in_counter_zone(cx: int, cy: int) -> bool:
    """Return True if the point (cx, cy) lies inside the counter exclusion zone."""
    if COUNTER_ZONE is None:
        return False
    zx1, zy1, zx2, zy2 = COUNTER_ZONE
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


# ── Video setup ───────────────────────────────────────────────────────────────
# KEDIT1: Input file path
video_path = "cafe.qt"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# ── Load pose model ───────────────────────────────────────────────────────────
pose_model = YOLO("yolo11n-pose.pt")

# ── Output setup ──────────────────────────────────────────────────────────────
output_dir = "runs/detect/track_queue"
os.makedirs(output_dir, exist_ok=True)
# KEDIT2: Writing output file path
output_path = os.path.join(output_dir, "cafe_output.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

print(f"Starting queue tracking... Output will be saved to {output_path}")

# ── Frame-by-frame processing ─────────────────────────────────────────────────
tracking_data = []
frame_count   = 0

# Maps track_id → frame number when they were first detected (queue entry order)
entry_order: dict[int, int] = {}

# Persisted queue direction vector (dx, dy) – unit vector from front → back.
# None until a valid queue (≥2 people) is first observed.
queue_direction: tuple[float, float] | None = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run pose estimation with persistent tracking (persons only: class 0)
    results   = pose_model.track(frame, persist=True, classes=[0], verbose=False)

    queue_candidates = []

    boxes_res = results[0].boxes
    kps_res   = results[0].keypoints

    # ── Determine which detections are standing BEFORE plotting ───────────
    standing_indices = []
    if boxes_res is not None and kps_res is not None and boxes_res.id is not None:
        for i, kps in enumerate(kps_res.data):
            if is_standing(kps):
                standing_indices.append(i)

    # Plot only the standing detections so sitting people never appear
    if standing_indices:
        annotated_frame = results[0][standing_indices].plot()
    else:
        annotated_frame = frame.copy()

    if boxes_res is not None and kps_res is not None and boxes_res.id is not None:
        for i, (box, track_id, kps) in enumerate(zip(boxes_res, boxes_res.id, kps_res.data)):

            obj_id = int(track_id.item())

            # ── Pose-based standing filter ──────────────────────────────────
            if not is_standing(kps):
                continue  # already excluded from the plot above

            # ── Extract bounding-box geometry ───────────────────────────────
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ── Counter exclusion zone filter ───────────────────────────────
            if in_counter_zone(cx, cy):
                # Still draw the (cx,cy) label so coordinates are visible for tuning
                cv2.putText(annotated_frame, f"[STAFF] ({cx},{cy})",
                            (cx - 40, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 2)
                continue  # skip – do not add to queue_candidates

            # (bounding-box geometry already extracted above)

            # Record the first frame this ID was ever seen (= queue entry order)
            if obj_id not in entry_order:
                entry_order[obj_id] = frame_count

            queue_candidates.append({
                "id":    obj_id,
                "cx":    cx,
                "cy":    cy,
                "box":   (x1, y1, x2, y2),
                "entry": entry_order[obj_id],
            })

            tracking_data.append({
                "frame":     frame_count,
                "object_id": obj_id,
                "x":         cx,
                "y":         cy,
            })

            # Draw centre coordinate label
            cv2.putText(annotated_frame, f"({cx},{cy})",
                        (cx - 30, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    # ── Queue clustering ──────────────────────────────────────────────────────
    if queue_candidates:
        clusters = []
        for person in queue_candidates:
            ph = person["box"][3] - person["box"][1]  # person height (pixels)
            assigned = []

            for i, cluster in enumerate(clusters):
                for comp in cluster:
                    dist = math.hypot(person["cx"] - comp["cx"],
                                      person["cy"] - comp["cy"])
                    ch = comp["box"][3] - comp["box"][1]
                    if dist < 1.8 * max(ph, ch):
                        assigned.append(i)
                        break

            if not assigned:
                clusters.append([person])
            else:
                first = assigned[0]
                clusters[first].append(person)
                for other in sorted(assigned[1:], reverse=True):
                    clusters[first].extend(clusters[other])
                    del clusters[other]

        # Only consider groups with more than one person as a queue
        valid_queues = [c for c in clusters if len(c) > 1]

        if valid_queues:
            # Pick the largest queue
            main_queue  = max(valid_queues, key=len)
            # Sort by entry order: earliest entrant = front of queue,
            # latest entrant = back of queue (queue-end)
            sorted_queue = sorted(main_queue, key=lambda p: p["entry"])
            last_person  = sorted_queue[-1]  # person who joined most recently

            qx1, qy1, qx2, qy2 = last_person["box"]
            last_w = qx2 - qx1
            last_h = qy2 - qy1

            # ── Queue direction vector ────────────────────────────────────
            # Locked once on the first frame a valid queue (≥2 people) is seen.
            # Uses the full queue span (front → back) to establish the angle.
            # After that, direction NEVER changes — only last_person's live
            # position moves, keeping the "Spot behind" marker correctly angled.
            if queue_direction is None:
                if len(sorted_queue) >= 2:
                    ref_person = sorted_queue[0]
                    raw_dx = last_person["cx"] - ref_person["cx"]
                    raw_dy = last_person["cy"] - ref_person["cy"]
                    length = math.hypot(raw_dx, raw_dy)
                    queue_direction = (raw_dx / length, raw_dy / length) if length > 0 else (0.0, 1.0)
                else:
                    queue_direction = (0.0, 1.0)  # default: straight down
            # queue_direction is intentionally not updated after first detection

            dx, dy = queue_direction

            # ── "Spot behind" calculation ─────────────────────────────────
            distance = max(last_h, 50)
            spot_cx  = int(last_person["cx"] + dx * distance)
            spot_cy  = int(last_person["cy"] + dy * distance)

            sx1 = int(spot_cx - last_w / 2)
            sy1 = int(spot_cy - last_h / 2)
            sx2 = int(spot_cx + last_w / 2)
            sy2 = int(spot_cy + last_h / 2)

            # Draw target spot (green, thick)
            cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 4)
            cv2.putText(annotated_frame, "Spot behind",
                        (sx1, max(0, sy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Highlight last person in queue (red)
            cv2.rectangle(annotated_frame, (qx1, qy1), (qx2, qy2), (0, 0, 255), 2)

    # ── Draw counter exclusion zone overlay ───────────────────────────────────
    if COUNTER_ZONE is not None:
        zx1, zy1, zx2, zy2 = COUNTER_ZONE
        cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
        cv2.putText(annotated_frame, "Counter zone",
                    (zx1 + 4, zy1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # ── Write frame ───────────────────────────────────────────────────────────
    out.write(annotated_frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete! Exporting JSON...")
json_path = os.path.join(output_dir, "positional_data.json")
with open(json_path, "w") as f:
    json.dump(tracking_data, f, indent=4)
print(f"Queue tracking data saved to {json_path}")
