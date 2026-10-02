import cv2

from app.tracker import FaceTracker
from app.recognizer import FaceRecognizer
from app.face_registry import FaceRegistry
from app.database import Database
from app.logger import EventLogger
from app.event_manager import EventManager


class VideoProcessor:

    def __init__(self, config):

        self.config = config

        # ==================================================
        # CONFIGURATION
        # ==================================================

        detection_config = config[
            "detection"
        ]

        self.frame_skip = int(
            detection_config[
                "frame_skip"
            ]
        )

        confidence_threshold = float(
            detection_config[
                "confidence_threshold"
            ]
        )

        recognition_config = config[
            "recognition"
        ]

        similarity_threshold = float(
            recognition_config[
                "similarity_threshold"
            ]
        )

        tracking_config = config[
            "tracking"
        ]

        max_lost_frames = int(
            tracking_config[
                "max_lost_frames"
            ]
        )

        database_config = config[
            "database"
        ]

        database_path = database_config[
            "path"
        ]

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

        # ==================================================
        # MODEL
        # ==================================================

        model_path = (
            "models/yolov8n-face.pt"
        )

        # ==================================================
        # TRACKER
        # ==================================================

        print(
            "Loading face tracker..."
        )

        self.tracker = FaceTracker(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        # ==================================================
        # INSIGHTFACE
        # ==================================================

        print(
            "Loading InsightFace..."
        )

        self.recognizer = FaceRecognizer()

        # ==================================================
        # FACE REGISTRY
        # ==================================================

        print(
            "Loading face registry..."
        )

        self.registry = FaceRegistry(
            similarity_threshold=similarity_threshold
        )

        # ==================================================
        # DATABASE
        # ==================================================

        print(
            "Loading database..."
        )

        self.database = Database(
            database_path
        )

        # ==================================================
        # EVENT LOGGER
        # ==================================================

        print(
            "Loading event logger..."
        )

        self.logger = EventLogger(
            event_log
        )

        # ==================================================
        # EVENT MANAGER
        # ==================================================

        print(
            "Loading event manager..."
        )

        self.event_manager = EventManager(
            database=self.database,
            logger=self.logger,
            entry_directory=entry_directory,
            exit_directory=exit_directory,
            max_lost_frames=max_lost_frames
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        print(
            "All components loaded."
        )

    # ======================================================
    # FACE CROP
    # ======================================================

    def _crop_face(
        self,
        frame,
        bbox
    ):
        """
        Safely crop a face from the frame.
        """

        x1, y1, x2, y2 = bbox

        height, width = (
            frame.shape[:2]
        )

        # --------------------------------------------------
        # Clamp coordinates
        # --------------------------------------------------

        x1 = max(
            0,
            min(
                int(x1),
                width - 1
            )
        )

        y1 = max(
            0,
            min(
                int(y1),
                height - 1
            )
        )

        x2 = max(
            0,
            min(
                int(x2),
                width
            )
        )

        y2 = max(
            0,
            min(
                int(y2),
                height
            )
        )

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        face_crop = frame[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            return None

        return face_crop

    # ======================================================
    # PROCESS VIDEO
    # ======================================================

    def process_video(
        self,
        video_source=None
    ):
        """
        Process one video.

        Pipeline:

        Video
          ↓
        OpenCV
          ↓
        ByteTrack
          ↓
        Face Crop
          ↓
        InsightFace
          ↓
        Face Registry
          ↓
        Event Manager
          ↓
        SQLite + Logs + Images
        """

        if video_source is None:

            video_source = (
                self.config[
                    "video_source"
                ]
            )

        print()
        print(
            "=" * 60
        )
        print(
            "VIDEO PROCESSING STARTED"
        )
        print(
            "=" * 60
        )

        print(
            f"Source: {video_source}"
        )

        print()

        # ==================================================
        # OPEN VIDEO
        # ==================================================

        cap = cv2.VideoCapture(
            video_source
        )

        if not cap.isOpened():

            print(
                "ERROR: Could not open video:"
            )

            print(
                video_source
            )

            return False

        # ==================================================
        # VIDEO INFORMATION
        # ==================================================

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        print(
            f"Resolution: "
            f"{width}x{height}"
        )

        print(
            f"FPS: {fps:.2f}"
        )

        print(
            f"Total frames: "
            f"{total_frames}"
        )

        print(
            f"Frame skip: "
            f"{self.frame_skip}"
        )

        print()

        # ==================================================
        # COUNTERS
        # ==================================================

        frame_number = 0

        processed_frames = 0

        # ==================================================
        # MAIN LOOP
        # ==================================================

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            frame_number += 1

            # --------------------------------------------------
            # FRAME SKIPPING
            # --------------------------------------------------

            if (
                frame_number
                % self.frame_skip
                != 0
            ):
                continue

            processed_frames += 1

            # --------------------------------------------------
            # TRACK FACES
            # --------------------------------------------------

            tracks = (
                self.tracker.track(
                    frame
                )
            )

            visible_track_ids = []

            # --------------------------------------------------
            # PROCESS TRACKS
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

                # --------------------------------------------------
                # CROP FACE
                # --------------------------------------------------

                face_crop = (
                    self._crop_face(
                        frame,
                        bbox
                    )
                )

                if face_crop is None:

                    continue

                # --------------------------------------------------
                # GENERATE EMBEDDING
                # --------------------------------------------------

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

                # --------------------------------------------------
                # IDENTIFY / REGISTER
                # --------------------------------------------------

                (
                    person_id,
                    is_new,
                    similarity
                ) = (
                    self.registry.identify_or_register(
                        embedding
                    )
                )

                # --------------------------------------------------
                # STATUS
                # --------------------------------------------------

                if is_new:

                    status = (
                        "REGISTERED"
                    )

                else:

                    status = (
                        "RECOGNIZED"
                    )

                # --------------------------------------------------
                # CONSOLE OUTPUT
                # --------------------------------------------------

                print(
                    f"Frame {frame_number}: "
                    f"Track {track_id} -> "
                    f"{person_id} "
                    f"{status} "
                    f"(similarity="
                    f"{similarity:.3f})"
                )

                # --------------------------------------------------
                # EVENT MANAGER
                # --------------------------------------------------

                self.event_manager.process_detection(
                    person_id=person_id,
                    track_id=track_id,
                    face_crop=face_crop,
                    similarity=similarity,
                    is_new=is_new
                )

            # --------------------------------------------------
            # UPDATE EVENT MANAGER
            # --------------------------------------------------

            self.event_manager.update_frame(
                current_frame_number=frame_number,
                visible_track_ids=visible_track_ids
            )

            # --------------------------------------------------
            # PROGRESS
            # --------------------------------------------------

            if (
                processed_frames
                % 10
                == 0
            ):

                print(
                    f"Processed frames: "
                    f"{processed_frames} | "
                    f"Video frame: "
                    f"{frame_number}"
                )

        # ==================================================
        # VIDEO FINISHED
        # ==================================================

        # Generate EXIT events for anyone still active.
        self.event_manager.flush_remaining_tracks(
            current_frame_number=frame_number
        )

        cap.release()

        # ==================================================
        # RESULTS
        # ==================================================

        unique_visitors = (
            self.database.get_unique_visitor_count()
        )

        print()
        print(
            "=" * 60
        )
        print(
            "VIDEO PROCESSING FINISHED"
        )
        print(
            "=" * 60
        )

        print(
            f"Total frames: "
            f"{total_frames}"
        )

        print(
            f"Processed frames: "
            f"{processed_frames}"
        )

        print(
            f"Unique visitors: "
            f"{unique_visitors}"
        )

        print()

        return True