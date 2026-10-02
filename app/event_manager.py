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

        self.entry_directory = Path(
            entry_directory
        )

        self.exit_directory = Path(
            exit_directory
        )

        self.entry_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.exit_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.max_lost_frames = max_lost_frames

        # --------------------------------------------------
        # Track-level state
        #
        # track_id -> {
        #     person_id,
        #     last_seen_frame,
        #     face_crop
        # }
        # --------------------------------------------------

        self.active_tracks = {}

        # --------------------------------------------------
        # Person-level state
        #
        # person_id -> {
        #     track_ids,
        #     last_seen_frame,
        #     face_crop
        # }
        #
        # This is important because ByteTrack can change
        # track IDs while the same person is still present.
        # --------------------------------------------------

        self.active_people = {}

        # People that have been registered at least once
        self.known_people = set()

    # ======================================================
    # TIME HELPERS
    # ======================================================

    def _timestamp(self):
        return datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def _filename_timestamp(self):
        return datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

    # ======================================================
    # IMAGE SAVING
    # ======================================================

    def _save_face_image(
        self,
        directory,
        person_id,
        face_crop
    ):
        """
        Safely save a face crop.

        Returns:
            str: saved image path
            None: if image could not be saved
        """

        if face_crop is None:
            return None

        if not hasattr(
            face_crop,
            "size"
        ):
            return None

        if face_crop.size == 0:
            return None

        image_timestamp = (
            self._filename_timestamp()
        )

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

    # ======================================================
    # PROCESS DETECTION
    # ======================================================

    def process_detection(
        self,
        person_id,
        track_id,
        face_crop,
        similarity,
        is_new
    ):
        """
        Process one recognized face.

        Important behavior:

        New person:
            PERSON_001 -> ENTRY

        Same person with same track:
            PERSON_001 -> recognition only

        Same person with new ByteTrack ID:
            PERSON_001 -> recognition only
            NO duplicate ENTRY

        If person completely disappears:
            PERSON_001 -> EXIT
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
        # Embedding log
        # --------------------------------------------------

        self.logger.embedding(
            person_id
        )

        # --------------------------------------------------
        # Tracking log
        # --------------------------------------------------

        self.logger.tracking(
            person_id,
            track_id
        )

        # --------------------------------------------------
        # Prepare face crop
        # --------------------------------------------------

        valid_crop = (
            face_crop is not None
            and hasattr(face_crop, "size")
            and face_crop.size > 0
        )

        latest_crop = (
            face_crop.copy()
            if valid_crop
            else None
        )

        # ==================================================
        # PERSON ALREADY ACTIVE
        # ==================================================

        if person_id in self.active_people:

            person_data = (
                self.active_people[
                    person_id
                ]
            )

            # Add the current ByteTrack ID
            person_data[
                "track_ids"
            ].add(track_id)

            # Update latest face crop
            if latest_crop is not None:

                person_data[
                    "face_crop"
                ] = latest_crop

            # Update person last-seen frame later
            # through update_frame()

        # ==================================================
        # NEW ACTIVE PERSON
        # ==================================================

        else:

            # --------------------------------------------------
            # Save ENTRY image
            # --------------------------------------------------

            saved_image_path = (
                self._save_face_image(
                    self.entry_directory,
                    person_id,
                    latest_crop
                )
            )

            # --------------------------------------------------
            # Database ENTRY
            # --------------------------------------------------

            self.database.add_visit_event(
                person_id=person_id,
                event_type="ENTRY",
                timestamp=timestamp,
                image_path=saved_image_path,
                track_id=track_id
            )

            # --------------------------------------------------
            # Log ENTRY
            # --------------------------------------------------

            self.logger.entry(
                person_id,
                track_id,
                saved_image_path
            )

            # --------------------------------------------------
            # Create active person
            # --------------------------------------------------

            self.active_people[
                person_id
            ] = {
                "track_ids": {
                    track_id
                },
                "last_seen_frame": None,
                "face_crop": latest_crop
            }

        # ==================================================
        # CREATE / UPDATE TRACK STATE
        # ==================================================

        self.active_tracks[
            track_id
        ] = {
            "person_id": person_id,
            "last_seen_frame": None,
            "face_crop": latest_crop
        }

    # ======================================================
    # UPDATE FRAME
    # ======================================================

    def update_frame(
        self,
        current_frame_number,
        visible_track_ids
    ):
        """
        Update tracking state.

        A track is considered lost after max_lost_frames.

        A person exits only when ALL of that person's
        ByteTrack IDs have disappeared.
        """

        visible_track_ids = set(
            visible_track_ids
        )

        # --------------------------------------------------
        # Update visible tracks
        # --------------------------------------------------

        for track_id in visible_track_ids:

            if track_id not in self.active_tracks:
                continue

            track_data = (
                self.active_tracks[
                    track_id
                ]
            )

            track_data[
                "last_seen_frame"
            ] = current_frame_number

            person_id = (
                track_data[
                    "person_id"
                ]
            )

            # --------------------------------------------------
            # Update person-level state
            # --------------------------------------------------

            if person_id in self.active_people:

                person_data = (
                    self.active_people[
                        person_id
                    ]
                )

                person_data[
                    "last_seen_frame"
                ] = current_frame_number

                # Update latest crop if available
                if (
                    track_data[
                        "face_crop"
                    ] is not None
                ):
                    person_data[
                        "face_crop"
                    ] = track_data[
                        "face_crop"
                    ]

        # --------------------------------------------------
        # Find tracks that disappeared
        # --------------------------------------------------

        tracks_to_remove = []

        for (
            track_id,
            track_data
        ) in self.active_tracks.items():

            # Currently visible
            if track_id in visible_track_ids:
                continue

            last_seen = (
                track_data[
                    "last_seen_frame"
                ]
            )

            # No previous frame recorded
            if last_seen is None:

                track_data[
                    "last_seen_frame"
                ] = current_frame_number

                continue

            lost_frames = (
                current_frame_number
                - last_seen
            )

            if (
                lost_frames
                >= self.max_lost_frames
            ):

                tracks_to_remove.append(
                    track_id
                )

        # --------------------------------------------------
        # Remove expired tracks
        # --------------------------------------------------

        for track_id in tracks_to_remove:

            if track_id not in self.active_tracks:
                continue

            track_data = (
                self.active_tracks[
                    track_id
                ]
            )

            person_id = (
                track_data[
                    "person_id"
                ]
            )

            # Remove track
            del self.active_tracks[
                track_id
            ]

            # --------------------------------------------------
            # Update person-level state
            # --------------------------------------------------

            if person_id not in self.active_people:
                continue

            person_data = (
                self.active_people[
                    person_id
                ]
            )

            person_data[
                "track_ids"
            ].discard(
                track_id
            )

            # --------------------------------------------------
            # IMPORTANT:
            #
            # If another ByteTrack ID still represents the
            # same person, DO NOT create EXIT.
            # --------------------------------------------------

            if person_data[
                "track_ids"
            ]:
                continue

            # --------------------------------------------------
            # No tracks remain.
            #
            # Therefore the person has exited.
            # --------------------------------------------------

            self._create_exit_event(
                person_id=person_id,
                person_data=person_data,
                track_id=track_id
            )

            # Remove active person
            del self.active_people[
                person_id
            ]

    # ======================================================
    # CREATE EXIT EVENT
    # ======================================================

    def _create_exit_event(
        self,
        person_id,
        person_data,
        track_id=None
    ):
        """
        Create exactly one EXIT event.
        """

        face_crop = (
            person_data.get(
                "face_crop"
            )
        )

        timestamp = self._timestamp()

        # --------------------------------------------------
        # Save exit image
        # --------------------------------------------------

        saved_image_path = (
            self._save_face_image(
                self.exit_directory,
                person_id,
                face_crop
            )
        )

        # --------------------------------------------------
        # Save EXIT event
        # --------------------------------------------------

        self.database.add_visit_event(
            person_id=person_id,
            event_type="EXIT",
            timestamp=timestamp,
            image_path=saved_image_path,
            track_id=track_id
        )

        # --------------------------------------------------
        # Log EXIT
        # --------------------------------------------------

        self.logger.exit(
            person_id,
            track_id,
            saved_image_path
        )

    # ======================================================
    # FLUSH AT END OF VIDEO
    # ======================================================

    def flush_remaining_tracks(
        self,
        current_frame_number
    ):
        """
        Generate EXIT events for people who are still
        active when the video ends.

        Without this method, a person appearing until the
        final video frame might never reach the normal
        max_lost_frames timeout.
        """

        active_person_ids = list(
            self.active_people.keys()
        )

        for person_id in active_person_ids:

            if person_id not in self.active_people:
                continue

            person_data = (
                self.active_people[
                    person_id
                ]
            )

            track_ids = (
                person_data.get(
                    "track_ids",
                    set()
                )
            )

            track_id = (
                next(iter(track_ids))
                if track_ids
                else None
            )

            self._create_exit_event(
                person_id=person_id,
                person_data=person_data,
                track_id=track_id
            )

        # Clear state
        self.active_tracks.clear()
        self.active_people.clear()