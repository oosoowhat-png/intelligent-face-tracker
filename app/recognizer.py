import cv2
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
        """
        Generate a 512-dimensional face embedding
        from a YOLO face crop.

        The YOLO crop can be relatively small, so it is
        enlarged before InsightFace processes it.
        """

        if face_image is None:
            return None

        if face_image.size == 0:
            return None

        height, width = face_image.shape[:2]

        if height < 20 or width < 20:
            return None

        # Enlarge small YOLO face crops so that
        # InsightFace's internal detector can identify
        # the face reliably.
        scale = 3

        enlarged = cv2.resize(
            face_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Add padding around the face.
        # This gives InsightFace's detector some context.
        padding = 40

        padded = cv2.copyMakeBorder(
            enlarged,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        faces = self.app.get(padded)

        if not faces:
            return None

        # Select the largest detected face.
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