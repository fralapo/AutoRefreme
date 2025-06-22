"""Utilities for face and object detection."""
from typing import List, Tuple, Dict, Any, Set
import cv2
from pathlib import Path
import requests
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from urllib.parse import urlparse
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

        # Define model directory
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load general object detection model (yolo11n.pt)
        self.obj_model = None
        obj_model_filename = "yolo11n.pt"
        obj_model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" # Corrected Ultralytics assets URL
        local_obj_model_path = model_dir / obj_model_filename

        if not local_obj_model_path.is_file():
            logger.info(f"'{obj_model_filename}' not found locally at {local_obj_model_path}. Attempting to download from {obj_model_url}...")
            try:
                response = requests.get(obj_model_url, stream=True)
                response.raise_for_status()
                with open(local_obj_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded '{obj_model_filename}' to {local_obj_model_path}")
            except Exception as e_download_obj:
                logger.error(f"Could not download '{obj_model_filename}' from {obj_model_url}: {e_download_obj}. Object detection might not work.", exc_info=True)
                # self.obj_model remains None

        if local_obj_model_path.is_file() and self.obj_model is None: # Attempt to load if downloaded or already exists
            try:
                self.obj_model = YOLO(str(local_obj_model_path))
                logger.info(f"Successfully loaded general object model from {local_obj_model_path}.")
            except Exception as e_load_obj:
                logger.error(f"Could not load general object model from {local_obj_model_path}: {e_load_obj}. Object detection might not work.", exc_info=True)
                self.obj_model = None # Ensure it's None if loading fails
        elif not local_obj_model_path.is_file():
            logger.error(f"General object model '{obj_model_filename}' not found at {local_obj_model_path} and download failed or was not attempted. Object detection will be unavailable.")


        # Attempt to load specialized YOLO face detection model (yolov11n-face.pt)
        self.yolo_face_model = None
        self.mp_face_model = None

        face_model_filename = "yolov11n-face.pt"
        # Define the URL for the face model on GitHub
        face_model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt"

        # Determine the local path where the model should be stored (now using model_dir)
        # model_dir is already defined and created above: Path("models")
        local_face_model_path = model_dir / face_model_filename # Changed from local_model_path

        # Check if the file exists locally
        if not local_face_model_path.is_file():
            logger.info(f"'{face_model_filename}' not found locally at {local_face_model_path}. Attempting to download from {face_model_url}...")
            try:
                # Download the file from the URL using requests
                # import requests # Already imported at the top of the file
                response = requests.get(face_model_url, stream=True)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                with open(local_face_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded '{face_model_filename}' to {local_face_model_path}")

            except Exception as e_download_face: # Renamed e_download to avoid conflict if in same scope later
                logger.warning(f"Could not download '{face_model_filename}' from {face_model_url}: {e_download_face}")
                # local_face_model_path remains valid path, but file won't exist if download failed

        # Attempt to load the YOLO face model from the local path (if determined and exists)
        if local_face_model_path.is_file(): # Check again if file exists (it might have been downloaded)
            try:
                logger.info(f"Attempting to load YOLO face model from {local_face_model_path}...")
                # Use str() to ensure compatibility with YOLO() which might expect a string path
                self.yolo_face_model = YOLO(str(local_face_model_path))
                logger.info(f"Successfully loaded YOLO face model from {local_face_model_path}.")
            except Exception as e_load_face: # Renamed e_load
                logger.warning(f"Could not load YOLO face model from {local_face_model_path}: {e_load_face}")
                self.yolo_face_model = None # Ensure model is None if loading fails
        else: # File does not exist (either wasn't there initially or download failed)
            logger.warning(f"YOLO face model file not found at {local_face_model_path}. Face detection with YOLO will be unavailable.")
            self.yolo_face_model = None


        # If self.yolo_face_model is still None after attempts, fallback to MediaPipe
        if self.yolo_face_model is None:
            logger.warning(f"YOLO face model not loaded. Falling back to MediaPipe for face detection.")
            try:
                self.mp_face_model = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=self.mp_face_conf
                )
                logger.info("MediaPipe Face Detection initialized as fallback.")
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
