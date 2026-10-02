import cv2

from app.tracker import FaceTracker


VIDEO_PATH = "sample/input.mp4"
MODEL_PATH = "models/yolov8n-face.pt"


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
saved_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    if frame_number % 5 != 0:
        continue

    tracks = tracker.track(frame)

    for track in tracks:

        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]

        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        crop_height, crop_width = face_crop.shape[:2]

        print(
            f"Frame {frame_number}: "
            f"Track {track_id} -> "
            f"crop size = {crop_width}x{crop_height}"
        )

        # Save first 10 crops
        if saved_count < 10:

            filename = (
                f"output/face_crop_"
                f"{saved_count + 1}_"
                f"track_{track_id}.jpg"
            )

            cv2.imwrite(
                filename,
                face_crop
            )

            print(
                f"Saved: {filename}"
            )

            saved_count += 1


cap.release()

print("\nCrop debugging finished.")