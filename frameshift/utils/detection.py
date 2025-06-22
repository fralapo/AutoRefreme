"""Utilities for face and object detection."""
from typing import List, Tuple, Dict, Any, Set
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger('frameshift.utils.detection')


class Detector:
    """
    Handles face and object detection.
    Uses a specialized YOLOv8 model for face detection (from Hugging Face)
    and a general YOLOv8n model for other objects (conditionally).
    MediaPipe is used as a fallback for face detection if the YOLO face model fails to load.
    """

    def __init__(self, yolo_face_conf: float = 0.3, yolo_obj_conf: float = 0.25, mp_face_conf: float = 0.5):
        self.yolo_face_conf = yolo_face_conf
        self.yolo_obj_conf = yolo_obj_conf
        self.mp_face_conf = mp_face_conf

        # Load general object detection model (YOLOv8n)
        try:
            self.obj_model = YOLO("yolov8n.pt")
            logger.info("Successfully loaded YOLOv8n for general object detection.")
        except Exception as e:
            logger.error(f"Could not load YOLOv8n object model: {e}", exc_info=True)
            self.obj_model = None

        # Attempt to load specialized YOLOv8 face detection model
        self.yolo_face_model = None
        self.mp_face_model = None # Will be initialized if YOLO face model fails

        try:
            logger.info("Attempting to download/load YOLOv8-Face-Detection model...")
            face_model_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection",
                filename="model.pt"
            )
            self.yolo_face_model = YOLO(face_model_path)
            logger.info("Successfully loaded YOLOv8-Face-Detection model.")
            # if self.yolo_face_model and hasattr(self.yolo_face_model, 'names'):
                # logger.debug(f"YOLOv8 Face Model class names: {self.yolo_face_model.names}")
        except Exception as e:
            logger.warning(f"Could not download/load YOLOv8-Face-Detection model: {e}. Will use MediaPipe for faces if available.", exc_info=True)
            self.yolo_face_model = None

        if self.yolo_face_model is None:
            logger.info("Falling back to MediaPipe for face detection.")
            try:
                self.mp_face_model = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=self.mp_face_conf
                )
                logger.info("MediaPipe Face Detection initialized.")
            except Exception as e_mp:
                logger.error(f"Could not initialize MediaPipe Face Detection: {e_mp}", exc_info=True)
                self.mp_face_model = None


    def detect(self, frame: np.ndarray, active_object_labels: Set[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return lists of face detections and object detections.
        Faces are always detected (YOLOv8-Face or MediaPipe fallback).
        Other objects are detected by YOLOv8n only if active_object_labels is not empty.
        Each detection is a dict: {'box': (x1,y1,x2,y2), 'label': str, 'confidence': float}
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For MediaPipe if used

        faces_detected: List[Dict[str, Any]] = []

        # 1. Face Detection
        if self.yolo_face_model:
            try:
                yolo_face_preds = self.yolo_face_model.predict(frame, imgsz=320, conf=self.yolo_face_conf, verbose=False)
                for res in yolo_face_preds:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confidences = res.boxes.conf.cpu().numpy()
                    # class_ids = res.boxes.cls.cpu().numpy().astype(int)
                    # Assuming class 0 is 'face' for this model, or it only has one class.
                    # For arnabdhar/YOLOv8-Face-Detection, it's typically class 0: 'face'.
                    # yolo_face_class_names = res.names if hasattr(res, 'names') and res.names else {0: 'face'}

                    for i in range(len(boxes)):
                        box_coords = tuple(boxes[i].astype(int))
                        # label = yolo_face_class_names.get(class_ids[i], 'face') # Ensure label is 'face'
                        faces_detected.append({
                            'box': box_coords,
                            'label': 'face', # Standardize label
                            'confidence': confidences[i]
                        })
            except Exception as e_yolo_face:
                logger.error(f"YOLOv8-Face-Detection predict failed: {e_yolo_face}. Attempting MediaPipe fallback if available.", exc_info=True)
                if self.mp_face_model:
                    self.yolo_face_model = None
                else:
                     faces_detected = []

        if not self.yolo_face_model and self.mp_face_model:
            try:
                mp_results = self.mp_face_model.process(rgb_frame)
                if mp_results.detections:
                    for det in mp_results.detections:
                        rel_box = det.location_data.relative_bounding_box
                        x1 = int(rel_box.xmin * w)
                        y1 = int(rel_box.ymin * h)
                        bw = int(rel_box.width * w)
                        bh = int(rel_box.height * h)
                        faces_detected.append({
                            'box': (x1, y1, x1 + bw, y1 + bh),
                            'label': 'face',
                            'confidence': det.score[0] if det.score else self.mp_face_conf
                        })
            except Exception as e_mp_face:
                 logger.error(f"MediaPipe face detection failed: {e_mp_face}", exc_info=True)
                 faces_detected = []


        # 2. Object Detection (Conditional)
        objects_detected: List[Dict[str, Any]] = []
        if self.obj_model and active_object_labels:
            try:
                yolo_obj_preds = self.obj_model.predict(frame, imgsz=320, conf=self.yolo_obj_conf, verbose=False)
                for res in yolo_obj_preds:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confidences = res.boxes.conf.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy().astype(int)
                    class_names_map = res.names if hasattr(res, 'names') and res.names else self.obj_model.names

                    for i in range(len(boxes)):
                        label = class_names_map.get(class_ids[i], f"class_{class_ids[i]}")
                        if label in active_object_labels: # Filter by active labels
                            box_coords = tuple(boxes[i].astype(int))
                            objects_detected.append({
                                'box': box_coords,
                                'label': label,
                                'confidence': confidences[i]
                            })
            except Exception as e_yolo_obj:
                logger.error(f"YOLOv8n object detection predict failed: {e_yolo_obj}", exc_info=True)
                objects_detected = []

        return faces_detected, objects_detected
