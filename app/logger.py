import logging
from pathlib import Path


class EventLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.logger = logging.getLogger(
            "face_tracker_events"
        )

        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if the logger
        # is initialized more than once.
        self.logger.handlers.clear()

        handler = logging.FileHandler(
            self.log_path,
            encoding="utf-8"
        )

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def log(self, event_type, message):
        self.logger.info(
            f"{event_type} | {message}"
        )

    def registration(self, person_id):
        self.log(
            "REGISTRATION",
            f"New face registered: {person_id}"
        )

    def embedding(self, person_id):
        self.log(
            "EMBEDDING",
            f"Embedding generated for {person_id}"
        )

    def recognition(
        self,
        person_id,
        similarity
    ):
        self.log(
            "RECOGNITION",
            f"{person_id} recognized "
            f"(similarity={similarity:.3f})"
        )

    def tracking(
        self,
        person_id,
        track_id
    ):
        self.log(
            "TRACKING",
            f"{person_id} tracked "
            f"(track_id={track_id})"
        )

    def entry(
        self,
        person_id,
        track_id,
        image_path
    ):
        self.log(
            "ENTRY",
            f"{person_id} entered "
            f"(track_id={track_id}, "
            f"image={image_path})"
        )

    def exit(
        self,
        person_id,
        track_id,
        image_path
    ):
        self.log(
            "EXIT",
            f"{person_id} exited "
            f"(track_id={track_id}, "
            f"image={image_path})"
        )