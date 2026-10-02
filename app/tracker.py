from ultralytics import YOLO


class FaceTracker:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            verbose=False
        )

        tracks = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            if boxes.id is None:
                continue

            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for track_id, bbox, confidence in zip(
                track_ids,
                coordinates,
                confidences
            ):
                x1, y1, x2, y2 = bbox

                tracks.append({
                    "track_id": int(track_id),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],
                    "confidence": float(confidence)
                })

        return tracks