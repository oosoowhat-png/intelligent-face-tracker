import cv2

from app.detector import FaceDetector


MODEL_PATH = "models/yolov8n-face.pt"
VIDEO_PATH = "sample/input.mp4"


detector = FaceDetector(
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

    detections = detector.detect(frame)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Face {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "YOLO Face Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Processed frames: {frame_number}")