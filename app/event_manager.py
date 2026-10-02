from datetime import datetime
from pathlib import Path

import cv2


class EventManager:
    def __init__(
        self,
        database,
        logger,
        entry_directory,
        exit_directory,
        max_lost_frames=30
    ):
        self.database = database
        self.logger = logger

        self.entry_directory = Path(entry_directory)
        self.exit_directory = Path(exit_directory)

        self.entry_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.exit_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.max_lost_frames = max_lost_frames

        # Active ByteTrack tracks
        self.active_tracks = {}

        # People who have been registered
        self.known_people = set()

    def _timestamp(self):
        return datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def _filename_timestamp(self):
        return datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

    def _save_face_image(
        self,
        directory,
        person_id,
        face_crop
    ):
        """
        Save a face crop safely.

        Returns:
            str path if image was successfully saved
            None if image is invalid or saving failed
        """

        if face_crop is None:
            return None

        if not hasattr(face_crop, "size"):
            return None

        if face_crop.size == 0:
            return None

        image_timestamp = self._filename_timestamp()

        image_path = (
            directory
            / f"{person_id}_{image_timestamp}.jpg"
        )

        success = cv2.imwrite(
            str(image_path),
            face_crop
        )

        if success:
            return str(image_path)

        return None

    def process_detection(
        self,
        person_id,
        track_id,
        face_crop,
        similarity,
        is_new
    ):
        """
        Process a recognized or newly registered person.

        Creates an ENTRY event only when the track is first seen.
        """

        timestamp = self._timestamp()

        # --------------------------------------------------
        # Registration / Recognition
        # --------------------------------------------------

        if is_new:
            self.database.add_visitor(
                person_id,
                timestamp
            )

            self.known_people.add(
                person_id
            )

            self.logger.registration(
                person_id
            )

        else:
            self.database.update_visitor(
                person_id,
                timestamp
            )

            self.logger.recognition(
                person_id,
                similarity
            )

        # --------------------------------------------------
        # Embedding and tracking logs
        # --------------------------------------------------

        self.logger.embedding(
            person_id
        )

        self.logger.tracking(
            person_id,
            track_id
        )

        # --------------------------------------------------
        # ENTRY
        # --------------------------------------------------

        if track_id not in self.active_tracks:

            saved_image_path = self._save_face_image(
                self.entry_directory,
                person_id,
                face_crop
            )

            # Save entry event in database
            self.database.add_visit_event(
                person_id=person_id,
                event_type="ENTRY",
                timestamp=timestamp,
                image_path=saved_image_path,
                track_id=track_id
            )

            # Write ENTRY to events.log
            self.logger.entry(
                person_id,
                track_id,
                saved_image_path
            )

            # Store active track
            self.active_tracks[track_id] = {
                "person_id": person_id,
                "last_seen_frame": None,
                "face_crop": (
                    face_crop.copy()
                    if (
                        face_crop is not None
                        and hasattr(face_crop, "size")
                        and face_crop.size > 0
                    )
                    else None
                )
            }

        else:

            # Existing track
            self.active_tracks[track_id][
                "person_id"
            ] = person_id

            # Update latest face crop only if valid
            if (
                face_crop is not None
                and hasattr(face_crop, "size")
                and face_crop.size > 0
            ):
                self.active_tracks[track_id][
                    "face_crop"
                ] = face_crop.copy()

    def update_frame(
        self,
        current_frame_number,
        visible_track_ids
    ):
        """
        Update tracking state.

        If a track has not been visible for max_lost_frames,
        an EXIT event is generated.
        """

        visible_track_ids = set(
            visible_track_ids
        )

        # --------------------------------------------------
        # Update last-seen frame for visible tracks
        # --------------------------------------------------

        for track_id in visible_track_ids:

            if track_id in self.active_tracks:

                self.active_tracks[track_id][
                    "last_seen_frame"
                ] = current_frame_number

        # --------------------------------------------------
        # Initialize last_seen_frame for new tracks
        # --------------------------------------------------

        for track_id in self.active_tracks:

            if (
                self.active_tracks[track_id][
                    "last_seen_frame"
                ] is None
            ):

                self.active_tracks[track_id][
                    "last_seen_frame"
                ] = current_frame_number

        # --------------------------------------------------
        # Find lost tracks
        # --------------------------------------------------

        tracks_to_remove = []

        for (
            track_id,
            track_data
        ) in self.active_tracks.items():

            # Track is currently visible
            if track_id in visible_track_ids:
                continue

            last_seen = track_data[
                "last_seen_frame"
            ]

            if last_seen is None:
                continue

            lost_frames = (
                current_frame_number
                - last_seen
            )

            # Track has disappeared long enough
            if lost_frames >= self.max_lost_frames:

                self._create_exit_event(
                    track_id,
                    track_data
                )

                tracks_to_remove.append(
                    track_id
                )

        # --------------------------------------------------
        # Remove exited tracks
        # --------------------------------------------------

        for track_id in tracks_to_remove:

            del self.active_tracks[
                track_id
            ]

    def _create_exit_event(
        self,
        track_id,
        track_data
    ):
        """
        Create exactly one EXIT event for a lost track.
        """

        person_id = track_data[
            "person_id"
        ]

        face_crop = track_data.get(
            "face_crop"
        )

        timestamp = self._timestamp()

        # --------------------------------------------------
        # Save exit face image
        # --------------------------------------------------

        saved_image_path = self._save_face_image(
            self.exit_directory,
            person_id,
            face_crop
        )

        # --------------------------------------------------
        # Save EXIT event in database
        # --------------------------------------------------

        self.database.add_visit_event(
            person_id=person_id,
            event_type="EXIT",
            timestamp=timestamp,
            image_path=saved_image_path,
            track_id=track_id
        )

        # --------------------------------------------------
        # Write EXIT to events.log
        # --------------------------------------------------

        self.logger.exit(
            person_id,
            track_id,
            saved_image_path
        )