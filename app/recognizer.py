import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_embedding(self, face_image):
        faces = self.app.get(face_image)

        if not faces:
            return None

        # Select the largest detected face
        face = max(
            faces,
            key=lambda item: (
                item.bbox[2] - item.bbox[0]
            ) * (
                item.bbox[3] - item.bbox[1]
            )
        )

        embedding = np.asarray(
            face.embedding,
            dtype=np.float32
        )

        return embedding