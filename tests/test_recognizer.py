import cv2

from app.recognizer import FaceRecognizer


VIDEO_PATH = "sample/input.mp4"


recognizer = FaceRecognizer()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

frame_number = 0
embedding_generated = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Test every 5th frame
    if frame_number % 5 != 0:
        continue

    embedding = recognizer.get_embedding(frame)

    if embedding is not None:
        print(
            f"Frame {frame_number}: "
            f"Embedding generated"
        )

        print(
            f"Embedding shape: {embedding.shape}"
        )

        print(
            f"Embedding dtype: {embedding.dtype}"
        )

        embedding_generated = True

        break


cap.release()

if embedding_generated:
    print("\nINSIGHTFACE TEST PASSED")
else:
    print("\nNO FACE EMBEDDING WAS GENERATED")