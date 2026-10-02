import numpy as np


class FaceRegistry:

    def __init__(
        self,
        similarity_threshold=0.45,
        database=None
    ):

        self.similarity_threshold = (
            similarity_threshold
        )

        self.database = database

        self.registered_faces = {}

        self.next_id = 1

        # --------------------------------------------------
        # Load persistent identities
        # --------------------------------------------------

        if self.database is not None:

            self.registered_faces = (
                self.database.get_all_embeddings()
            )

            self._update_next_id()

    # ======================================================
    # ID MANAGEMENT
    # ======================================================

    def _update_next_id(self):

        highest_id = 0

        for person_id in (
            self.registered_faces.keys()
        ):

            if not person_id.startswith(
                "PERSON_"
            ):
                continue

            try:

                number = int(
                    person_id.split("_")[1]
                )

                highest_id = max(
                    highest_id,
                    number
                )

            except (
                ValueError,
                IndexError
            ):

                continue

        self.next_id = (
            highest_id + 1
        )

    def _generate_face_id(self):

        face_id = (
            f"PERSON_{self.next_id:03d}"
        )

        self.next_id += 1

        return face_id

    # ======================================================
    # COSINE SIMILARITY
    # ======================================================

    def _cosine_similarity(
        self,
        embedding1,
        embedding2
    ):

        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )

        norm1 = np.linalg.norm(
            embedding1
        )

        norm2 = np.linalg.norm(
            embedding2
        )

        if norm1 == 0 or norm2 == 0:

            return 0.0

        return float(
            np.dot(
                embedding1,
                embedding2
            )
            /
            (
                norm1 * norm2
            )
        )

    # ======================================================
    # FIND MATCH
    # ======================================================

    def find_match(
        self,
        embedding
    ):

        if not self.registered_faces:

            return None, 0.0

        best_face_id = None

        best_similarity = 0.0

        for (
            face_id,
            registered_embedding
        ) in self.registered_faces.items():

            similarity = (
                self._cosine_similarity(
                    embedding,
                    registered_embedding
                )
            )

            if similarity > best_similarity:

                best_similarity = (
                    similarity
                )

                best_face_id = (
                    face_id
                )

        if (
            best_similarity
            >= self.similarity_threshold
        ):

            return (
                best_face_id,
                best_similarity
            )

        return (
            None,
            best_similarity
        )

    # ======================================================
    # REGISTER FACE
    # ======================================================

    def register_face(
        self,
        embedding
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        face_id = (
            self._generate_face_id()
        )

        # --------------------------------------------------
        # Memory
        # --------------------------------------------------

        self.registered_faces[
            face_id
        ] = embedding

        # --------------------------------------------------
        # Persistent database
        # --------------------------------------------------

        if self.database is not None:

            self.database.save_embedding(
                person_id=face_id,
                embedding=embedding
            )

        return face_id

    # ======================================================
    # IDENTIFY OR REGISTER
    # ======================================================

    def identify_or_register(
        self,
        embedding
    ):

        face_id, similarity = (
            self.find_match(
                embedding
            )
        )

        # --------------------------------------------------
        # Existing person
        # --------------------------------------------------

        if face_id is not None:

            return (
                face_id,
                False,
                similarity
            )

        # --------------------------------------------------
        # New person
        # --------------------------------------------------

        face_id = (
            self.register_face(
                embedding
            )
        )

        return (
            face_id,
            True,
            1.0
        )

    # ======================================================
    # INFORMATION
    # ======================================================

    def get_registered_count(self):

        return len(
            self.registered_faces
        )