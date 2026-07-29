'''from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/vehicle_yolo_v1/weights/best.pt")

# Open your video or CCTV stream
cap = cv2.VideoCapture("test/test-3.mp4")  # or 0 for webcam

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,           # Keep track IDs consistent across frames
        tracker="bytetrack.yaml",
        conf=0.3,               # base confidence threshold
        iou=0.5
    )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            if conf < 0.5:
                continue  # skip low-confidence detections

            x1, y1, x2, y2 = map(int, box)

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # prepare probability text (e.g., "0.87")
            prob_text = f"{conf:.2f}"

            # draw text above the box
            cv2.putText(
                frame,
                prob_text,
                (x1, y1 - 10),  # just above top-left corner of box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/vehicle_yolo_v1/weights/best.pt")

# Open your video or CCTV stream
cap = cv2.VideoCapture("test/test-1.mp4")

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

delay = int(1000 / fps)

# ─── SPEED ZONE CONFIG ─────────────────────────────────────
ZONE_X1, ZONE_Y1 = 500, 100      # top-left corner of zone (adjust for your video)
ZONE_X2, ZONE_Y2 = 919, 729      # bottom-right corner of zone
ZONE_DISTANCE_METERS = 800       # real-world distance this zone represents
# ───────────────────────────────────────────────────────────

vehicle_times = {}  # track_id -> {"in_zone": bool, "enter_time": float, "speed": float or None}

def in_zone(cx, cy):
    return ZONE_X1 <= cx <= ZONE_X2 and ZONE_Y1 <= cy <= ZONE_Y2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get video timestamp (use this instead of wall-clock time)
    now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Run detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.3,
        iou=0.5
    )

    # Draw speed zone on frame
    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)
    cv2.putText(
        frame,
        "Speed Zone",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Initialize entry if not present
            if track_id not in vehicle_times:
                vehicle_times[track_id] = {"in_zone": False, "enter_time": None, "speed": None}

            v = vehicle_times[track_id]
            currently_in_zone = in_zone(cx, cy)

            # ENTER EVENT
            if currently_in_zone and not v["in_zone"]:
                v["in_zone"] = True
                v["enter_time"] = now

            # EXIT EVENT
            if not currently_in_zone and v["in_zone"]:
                v["in_zone"] = False
                if v["enter_time"] is not None:
                    dt = now - v["enter_time"]
                    if dt > 0:
                        # speed in km/h
                        speed_kmh = (ZONE_DISTANCE_METERS / dt) * 3.6
                        v["speed"] = speed_kmh

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display speed or confidence
            if v["speed"] is not None:
                label = f"{v['speed']:.1f} km/h"
                color = (0, 255, 0)  # green
            else:
                label = f"{conf:.2f}"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Vehicle Speed Detection", frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

