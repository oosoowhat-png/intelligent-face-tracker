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

        # ==================================================
        # ACTIVE TRACKS
        #
        # track_id -> {
        #     person_id,
        #     last_seen_frame,
        #     face_crop
        # }
        # ==================================================

        self.active_tracks = {}

        # ==================================================
        # ACTIVE PEOPLE
        #
        # person_id -> {
        #     track_ids,
        #     last_seen_frame,
        #     face_crop
        # }
        #
        # A person can have multiple ByteTrack IDs during
        # one visit. We therefore track at PERSON level.
        # ==================================================

        self.active_people = {}

        # ==================================================
        # KNOWN PEOPLE
        #
        # Used only for runtime bookkeeping.
        # Persistent identity is handled by FaceRegistry
        # and SQLite embeddings.
        # ==================================================

        self.known_people = set()

    # ======================================================
    # TIME
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
    # SAVE FACE IMAGE
    # ======================================================

    def _save_face_image(
        self,
        directory,
        person_id,
        face_crop
    ):
        """
        Save a face crop as a JPEG.

        Returns:
            str: saved path
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

        timestamp = (
            self._filename_timestamp()
        )

        image_path = (
            directory
            / f"{person_id}_{timestamp}.jpg"
        )

        success = cv2.imwrite(
            str(image_path),
            face_crop
        )

        if not success:
            return None

        return str(image_path)

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
        Process one recognized face detection.

        Important:

        A new ByteTrack ID does NOT automatically create
        a new ENTRY.

        ENTRY is created only when the person is not
        currently active.

        Example:

            PERSON_001 / track 139
                -> ENTRY

            PERSON_001 / track 139
                -> recognition

            PERSON_001 / track 2
                -> recognition
                -> NO new ENTRY

            PERSON_001 disappears
                -> EXIT
        """

        timestamp = self._timestamp()

        # ==================================================
        # REGISTRATION / RECOGNITION
        # ==================================================

        if is_new:

            # New identity.
            #
            # record_entry() below will create the visitor
            # record with visit_count = 1.
            self.known_people.add(
                person_id
            )

            self.logger.registration(
                person_id
            )

        else:

            # Existing identity.
            self.known_people.add(
                person_id
            )

            self.database.update_visitor(
                person_id,
                timestamp
            )

            self.logger.recognition(
                person_id,
                similarity
            )

        # ==================================================
        # EMBEDDING EVENT
        # ==================================================

        self.logger.embedding(
            person_id
        )

        # ==================================================
        # TRACKING EVENT
        # ==================================================

        self.logger.tracking(
            person_id,
            track_id
        )

        # ==================================================
        # VALIDATE FACE CROP
        # ==================================================

        valid_crop = (
            face_crop is not None
            and hasattr(
                face_crop,
                "size"
            )
            and face_crop.size > 0
        )

        if valid_crop:

            latest_crop = (
                face_crop.copy()
            )

        else:

            latest_crop = None

        # ==================================================
        # PERSON ALREADY ACTIVE
        # ==================================================

        if person_id in self.active_people:

            person_data = (
                self.active_people[
                    person_id
                ]
            )

            # Add current ByteTrack ID
            person_data[
                "track_ids"
            ].add(
                track_id
            )

            # Update latest face crop
            if latest_crop is not None:

                person_data[
                    "face_crop"
                ] = latest_crop

        # ==================================================
        # NEW ACTIVE SESSION
        # ==================================================

        else:

            # --------------------------------------------------
            # This is a genuine new ENTRY/session.
            # --------------------------------------------------

            # Record visit.
            #
            # New person:
            #   visit_count = 1
            #
            # Returning person:
            #   visit_count += 1
            self.database.record_entry(
                person_id,
                timestamp
            )

            # --------------------------------------------------
            # Save entry image
            # --------------------------------------------------

            saved_image_path = (
                self._save_face_image(
                    self.entry_directory,
                    person_id,
                    latest_crop
                )
            )

            # --------------------------------------------------
            # Database ENTRY event
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

        A track becomes expired after max_lost_frames.

        A person receives EXIT only after ALL active
        ByteTrack IDs associated with that person have
        expired.
        """

        visible_track_ids = set(
            visible_track_ids
        )

        # ==================================================
        # UPDATE VISIBLE TRACKS
        # ==================================================

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

            if person_id in self.active_people:

                person_data = (
                    self.active_people[
                        person_id
                    ]
                )

                person_data[
                    "last_seen_frame"
                ] = current_frame_number

                # Update latest crop
                if (
                    track_data[
                        "face_crop"
                    ] is not None
                ):

                    person_data[
                        "face_crop"
                    ] = (
                        track_data[
                            "face_crop"
                        ]
                    )

        # ==================================================
        # FIND EXPIRED TRACKS
        # ==================================================

        expired_tracks = []

        for (
            track_id,
            track_data
        ) in self.active_tracks.items():

            # Track is visible
            if track_id in visible_track_ids:
                continue

            last_seen = (
                track_data[
                    "last_seen_frame"
                ]
            )

            # No previous timestamp.
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

                expired_tracks.append(
                    track_id
                )

        # ==================================================
        # REMOVE EXPIRED TRACKS
        # ==================================================

        for track_id in expired_tracks:

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
            # Person might already have disappeared.
            # --------------------------------------------------

            if person_id not in self.active_people:
                continue

            person_data = (
                self.active_people[
                    person_id
                ]
            )

            # Remove this track from the person's active tracks
            person_data[
                "track_ids"
            ].discard(
                track_id
            )

            # ==================================================
            # IMPORTANT
            #
            # If another track still belongs to this person,
            # they are still considered inside.
            # ==================================================

            if person_data[
                "track_ids"
            ]:

                continue

            # ==================================================
            # NO ACTIVE TRACKS REMAIN
            #
            # PERSON HAS EXITED.
            # ==================================================

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
        Create one EXIT event for a completed visit.
        """

        timestamp = self._timestamp()

        face_crop = (
            person_data.get(
                "face_crop"
            )
        )

        # ==================================================
        # SAVE EXIT IMAGE
        # ==================================================

        saved_image_path = (
            self._save_face_image(
                self.exit_directory,
                person_id,
                face_crop
            )
        )

        # ==================================================
        # DATABASE EXIT
        # ==================================================

        self.database.add_visit_event(
            person_id=person_id,
            event_type="EXIT",
            timestamp=timestamp,
            image_path=saved_image_path,
            track_id=track_id
        )

        # ==================================================
        # LOG EXIT
        # ==================================================

        self.logger.exit(
            person_id,
            track_id,
            saved_image_path
        )

    # ======================================================
    # END OF VIDEO
    # ======================================================

    def flush_remaining_tracks(
        self,
        current_frame_number
    ):
        """
        Create EXIT events for all people still active
        when the video ends.

        This is necessary because the final video frame
        does not give ByteTrack enough future frames to
        reach max_lost_frames.
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

            if track_ids:

                # Use one of the person's active
                # track IDs for the exit record.
                track_id = next(
                    iter(track_ids)
                )

            else:

                track_id = None

            self._create_exit_event(
                person_id=person_id,
                person_data=person_data,
                track_id=track_id
            )

        # ==================================================
        # CLEAR VIDEO-SPECIFIC STATE
        #
        # The persistent face identities remain in
        # FaceRegistry/SQLite, but ByteTrack state must
        # start fresh for the next video.
        # ==================================================

        self.active_tracks.clear()

        self.active_people.clear()