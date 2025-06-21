"""Utilities for face and object detection."""
from typing import List, Tuple
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


class Detector:
    """Wrapper combining MediaPipe face detection and YOLO object detection."""

    def __init__(self):
        self.face_model = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        # Load a small YOLOv8 model
        self.obj_model = YOLO("yolov8n.pt")

    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Return lists of face boxes and object boxes in absolute coordinates."""
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_model.process(rgb)

        faces = []
        if face_results.detections:
            for det in face_results.detections:
                box = det.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)
                faces.append((x1, y1, x1 + bw, y1 + bh))

        objs = []
        obj_results = self.obj_model.predict(frame, imgsz=320, device="cpu", conf=0.25, verbose=False)
        for r in obj_results:
            for box in r.boxes.xyxy.numpy():
                x1, y1, x2, y2 = box.astype(int)
                objs.append((x1, y1, x2, y2))
        return faces, objs
