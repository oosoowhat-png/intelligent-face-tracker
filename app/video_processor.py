import cv2

from app.detector import FaceDetector
from app.recognizer import FaceRecognizer
from app.tracker import FaceTracker
from app.face_registry import FaceRegistry
from app.database import Database
from app.logger import EventLogger
from app.event_manager import EventManager


class VideoProcessor:

    def __init__(self, config):

        self.config = config

        # --------------------------------------------------
        # Detection configuration
        # --------------------------------------------------

        detection_config = config["detection"]

        self.frame_skip = detection_config[
            "frame_skip"
        ]

        confidence_threshold = detection_config[
            "confidence_threshold"
        ]

        image_size = detection_config[
            "image_size"
        ]

        # --------------------------------------------------
        # Recognition configuration
        # --------------------------------------------------

        recognition_config = config[
            "recognition"
        ]

        similarity_threshold = recognition_config[
            "similarity_threshold"
        ]

        # --------------------------------------------------
        # Tracking configuration
        # --------------------------------------------------

        tracking_config = config[
            "tracking"
        ]

        max_lost_frames = tracking_config[
            "max_lost_frames"
        ]

        # --------------------------------------------------
        # Database configuration
        # --------------------------------------------------

        database_config = config[
            "database"
        ]

        database_path = database_config[
            "path"
        ]

        # --------------------------------------------------
        # Logging configuration
        # --------------------------------------------------

        logging_config = config[
            "logging"
        ]

        event_log = logging_config[
            "event_log"
        ]

        entry_directory = logging_config[
            "entry_directory"
        ]

        exit_directory = logging_config[
            "exit_directory"
        ]

        # --------------------------------------------------
        # YOLO model
        # --------------------------------------------------

        model_path = "models/yolov8n-face.pt"

        # --------------------------------------------------
        # Initialize components
        # --------------------------------------------------

        print("Loading face detector...")

        self.detector = FaceDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        print("Loading face tracker...")

        self.tracker = FaceTracker(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        print("Loading InsightFace...")

        self.recognizer = FaceRecognizer()

        print("Loading face registry...")

        self.registry = FaceRegistry(
            similarity_threshold=similarity_threshold
        )

        print("Loading database...")

        self.database = Database(
            database_path
        )

        print("Loading event logger...")

        self.logger = EventLogger(
            event_log
        )

        print("Loading event manager...")

        self.event_manager = EventManager(
            database=self.database,
            logger=self.logger,
            entry_directory=entry_directory,
            exit_directory=exit_directory,
            max_lost_frames=max_lost_frames
        )

        self.image_size = image_size

        print("All components loaded.")

    # ------------------------------------------------------
    # Face crop helper
    # ------------------------------------------------------

    def _crop_face(
        self,
        frame,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        height, width = frame.shape[:2]

        # Keep coordinates inside image
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = frame[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            return None

        return face_crop

    # ------------------------------------------------------
    # Process video
    # ------------------------------------------------------

    def process_video(
        self,
        video_source=None
    ):

        if video_source is None:
            video_source = self.config[
                "video_source"
            ]

        print()
        print("=" * 60)
        print("VIDEO PROCESSING STARTED")
        print("=" * 60)
        print(f"Source: {video_source}")
        print()

        cap = cv2.VideoCapture(
            video_source
        )

        if not cap.isOpened():

            print(
                f"ERROR: Could not open video: "
                f"{video_source}"
            )

            return

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            f"Total frames: {total_frames}"
        )

        print(
            f"FPS: {fps:.2f}"
        )

        print(
            f"Frame skip: {self.frame_skip}"
        )

        frame_number = 0

        processed_frames = 0

        # --------------------------------------------------
        # Main video loop
        # --------------------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # --------------------------------------------------
            # Frame skipping
            # --------------------------------------------------

            if (
                frame_number % self.frame_skip
                != 0
            ):
                continue

            processed_frames += 1

            # --------------------------------------------------
            # Track faces
            # --------------------------------------------------

            tracks = self.tracker.track(
                frame
            )

            visible_track_ids = []

            # --------------------------------------------------
            # Process each tracked face
            # --------------------------------------------------

            for track in tracks:

                track_id = track[
                    "track_id"
                ]

                bbox = track[
                    "bbox"
                ]

                confidence = track[
                    "confidence"
                ]

                visible_track_ids.append(
                    track_id
                )

                # ------------------------------------------
                # Crop face
                # ------------------------------------------

                face_crop = self._crop_face(
                    frame,
                    bbox
                )

                if face_crop is None:
                    continue

                # ------------------------------------------
                # Generate embedding
                # ------------------------------------------

                embedding = (
                    self.recognizer.get_embedding(
                        face_crop
                    )
                )

                if embedding is None:

                    print(
                        f"Frame {frame_number}: "
                        f"Track {track_id} -> "
                        f"embedding failed"
                    )

                    continue

                # ------------------------------------------
                # Identify / register
                # ------------------------------------------

                (
                    person_id,
                    is_new,
                    similarity
                ) = self.registry.identify_or_register(
                    embedding
                )

                # ------------------------------------------
                # Print result
                # ------------------------------------------

                status = (
                    "REGISTERED"
                    if is_new
                    else "RECOGNIZED"
                )

                print(
                    f"Frame {frame_number}: "
                    f"Track {track_id} -> "
                    f"{person_id} "
                    f"{status} "
                    f"(similarity="
                    f"{similarity:.3f})"
                )

                # ------------------------------------------
                # Event manager
                # ------------------------------------------

                self.event_manager.process_detection(
                    person_id=person_id,
                    track_id=track_id,
                    face_crop=face_crop,
                    similarity=similarity,
                    is_new=is_new
                )

            # --------------------------------------------------
            # Update lost tracks / EXIT events
            # --------------------------------------------------

            self.event_manager.update_frame(
                current_frame_number=frame_number,
                visible_track_ids=visible_track_ids
            )

            # --------------------------------------------------
            # Progress
            # --------------------------------------------------

            if processed_frames % 10 == 0:

                print(
                    f"Processed frames: "
                    f"{processed_frames} | "
                    f"Video frame: "
                    f"{frame_number}"
                )

        # --------------------------------------------------
        # Video finished
        # --------------------------------------------------

        cap.release()

        print()
        print("=" * 60)
        print("VIDEO PROCESSING FINISHED")
        print("=" * 60)

        print(
            f"Total frames: {total_frames}"
        )

        print(
            f"Processed frames: "
            f"{processed_frames}"
        )

        print(
            f"Unique visitors: "
            f"{self.database.get_unique_visitor_count()}"
        )

        print()