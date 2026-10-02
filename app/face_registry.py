import numpy as np


class FaceRegistry:
    def __init__(self, similarity_threshold=0.45):
        self.similarity_threshold = similarity_threshold

        # Stores:
        # {
        #     "PERSON_001": embedding,
        #     "PERSON_002": embedding
        # }
        self.registered_faces = {}

        self.next_id = 1

    def _generate_face_id(self):
        face_id = f"PERSON_{self.next_id:03d}"
        self.next_id += 1
        return face_id

    def _cosine_similarity(self, embedding1, embedding2):
        embedding1 = np.asarray(embedding1)
        embedding2 = np.asarray(embedding2)

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(
            np.dot(embedding1, embedding2)
            / (norm1 * norm2)
        )

    def find_match(self, embedding):
        if not self.registered_faces:
            return None, 0.0

        best_face_id = None
        best_similarity = 0.0

        for face_id, registered_embedding in self.registered_faces.items():

            similarity = self._cosine_similarity(
                embedding,
                registered_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_face_id = face_id

        if best_similarity >= self.similarity_threshold:
            return best_face_id, best_similarity

        return None, best_similarity

    def register_face(self, embedding):
        face_id = self._generate_face_id()

        self.registered_faces[face_id] = np.asarray(
            embedding,
            dtype=np.float32
        )

        return face_id

    def identify_or_register(self, embedding):
        face_id, similarity = self.find_match(embedding)

        if face_id is not None:
            return face_id, False, similarity

        face_id = self.register_face(embedding)

        return face_id, True, 1.0