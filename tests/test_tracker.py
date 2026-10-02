import cv2

from app.tracker import FaceTracker


MODEL_PATH = "models/yolov8n-face.pt"
VIDEO_PATH = "sample/input.mp4"


tracker = FaceTracker(
    model_path=MODEL_PATH,
    confidence_threshold=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    tracks = tracker.track(frame)

    for track in tracks:
        track_id = track["track_id"]
        x1, y1, x2, y2 = track["bbox"]
        confidence = track["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Track ID: {track_id} | {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "YOLO + ByteTrack",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print(f"Processed frames: {frame_number}")