import cv2

from app.recognizer import FaceRecognizer
from app.face_registry import FaceRegistry


VIDEO_PATH = "sample/input.mp4"


recognizer = FaceRecognizer()

registry = FaceRegistry(
    similarity_threshold=0.45
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

    # Process every 5th frame
    if frame_number % 5 != 0:
        continue

    embedding = recognizer.get_embedding(frame)

    if embedding is None:
        continue

    face_id, is_new, similarity = (
        registry.identify_or_register(
            embedding
        )
    )

    if is_new:
        print(
            f"Frame {frame_number}: "
            f"{face_id} REGISTERED"
        )
    else:
        print(
            f"Frame {frame_number}: "
            f"{face_id} RECOGNIZED "
            f"(similarity={similarity:.3f})"
        )


cap.release()

print("\nRecognition pipeline finished.")

print(
    "Unique registered faces:",
    len(registry.registered_faces)
)