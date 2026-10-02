import cv2

from app.detector import FaceDetector
from app.tracker import FaceTracker
from app.recognizer import FaceRecognizer
from app.face_registry import FaceRegistry


VIDEO_PATH = "sample/input.mp4"
MODEL_PATH = "models/yolov8n-face.pt"

CONFIDENCE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.45


detector = FaceDetector(
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD
)

tracker = FaceTracker(
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD
)

recognizer = FaceRecognizer()

registry = FaceRegistry(
    similarity_threshold=SIMILARITY_THRESHOLD
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

    # Configurable frame skip
    if frame_number % 5 != 0:
        continue

    # -------------------------------------------------
    # YOLO + ByteTrack
    # -------------------------------------------------

    tracks = tracker.track(frame)

    for track in tracks:

        track_id = track["track_id"]
        x1, y1, x2, y2 = track["bbox"]
        confidence = track["confidence"]

        # -------------------------------------------------
        # Keep bounding box inside image boundaries
        # -------------------------------------------------

        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # -------------------------------------------------
        # Crop face
        # -------------------------------------------------

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # -------------------------------------------------
        # Generate InsightFace embedding
        # -------------------------------------------------

        embedding = recognizer.get_embedding(face_crop)

        if embedding is None:
            print(
                f"Frame {frame_number}: "
                f"Track {track_id} - embedding failed"
            )
            continue

        # -------------------------------------------------
        # Recognize or register
        # -------------------------------------------------

        face_id, is_new, similarity = (
            registry.identify_or_register(
                embedding
            )
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        label = (
            f"{face_id} | Track {track_id} | "
            f"{similarity:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # Console output
        # -------------------------------------------------

        if is_new:

            print(
                f"Frame {frame_number}: "
                f"Track {track_id} -> "
                f"{face_id} REGISTERED"
            )

        else:

            print(
                f"Frame {frame_number}: "
                f"Track {track_id} -> "
                f"{face_id} RECOGNIZED "
                f"(similarity={similarity:.3f})"
            )

    # -------------------------------------------------
    # Show video
    # -------------------------------------------------

    cv2.imshow(
        "Intelligent Face Tracker",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


print("\n====================================")
print("FULL PIPELINE TEST FINISHED")
print("====================================")
print(
    "Unique registered faces:",
    len(registry.registered_faces)
)