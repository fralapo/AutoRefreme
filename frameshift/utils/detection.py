"""Utilities for face and object detection."""
from typing import List, Tuple, Dict, Any
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
        # Ensure the model's class names are loaded if needed later, though predict populates them in results.
        # For direct access: self.yolo_class_names = self.obj_model.names

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return lists of face detections and object detections.
        Each detection is a dict: {'box': (x1,y1,x2,y2), 'label': str, 'confidence': float}
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        face_results_mp = self.face_model.process(rgb)
        faces_detected: List[Dict[str, Any]] = []
        if face_results_mp.detections:
            for det in face_results_mp.detections:
                rel_box = det.location_data.relative_bounding_box
                x1 = int(rel_box.xmin * w)
                y1 = int(rel_box.ymin * h)
                bw = int(rel_box.width * w)
                bh = int(rel_box.height * h)
                faces_detected.append({
                    'box': (x1, y1, x1 + bw, y1 + bh),
                    'label': 'face',
                    'confidence': det.score[0] if det.score else 0.5 # MediaPipe score is a list
                })

        # Object detection
        objects_detected: List[Dict[str, Any]] = []
        # Predict with YOLO. `verbose=False` to reduce console output.
        # imgsz can be tuned for speed vs accuracy.
        yolo_preds = self.obj_model.predict(frame, imgsz=320, device="cpu", conf=0.25, verbose=False)

        for res in yolo_preds: # Iterate over results for each image (though we pass one)
            boxes = res.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            confidences = res.boxes.conf.cpu().numpy() # Confidence scores
            class_ids = res.boxes.cls.cpu().numpy().astype(int) # Class IDs

            # Get class names from the model
            # self.obj_model.names should be like {0: 'person', 1: 'bicycle', ...}
            class_names_map = res.names if hasattr(res, 'names') and res.names else self.obj_model.names

            for i in range(len(boxes)):
                box_coords = tuple(boxes[i].astype(int))
                objects_detected.append({
                    'box': box_coords,
                    'label': class_names_map.get(class_ids[i], f"class_{class_ids[i]}"), # Get name or use raw class_id
                    'confidence': confidences[i]
                })

        return faces_detected, objects_detected
