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

        # Load general object detection model (YOLOv8n) - Ultralytics handles download
        try:
            self.obj_model = YOLO("yolov8n.pt")
            logger.info("Successfully loaded/initialized YOLOv8n for general object detection.")
        except Exception as e:
            logger.error(f"Could not load YOLOv8n object model: {e}", exc_info=True)
            self.obj_model = None

        # Attempt to load specialized YOLO face detection model (yolov11n-face.pt)
        self.yolo_face_model = None
        self.mp_face_model = None

        face_model_filename = "yolov11n-face.pt"
        # Corretto il nome del file come da tua ultima indicazione.
        # Questo URL è un asset diretto, attempt_download_asset è più per i release di ultralytics.
        # Per un URL diretto, potremmo usare un'altra utility o hf_hub_download se il modello fosse su HF.
        # Per ora, assumiamo che attempt_download_asset possa prendere un URL completo o che il file
        # sia gestito in modo simile a yolov8n.pt (cioè, specificando solo il nome se è un modello noto a ultralytics).
        # Se "yolov11n-face.pt" non è un asset che YOLO() o attempt_download_asset conosce,
        # dobbiamo implementare un download manuale o chiedere all'utente di scaricarlo.
        # Tentativo con attempt_download_asset per un file specifico da un URL:
        # Questo richiede che il file sia un asset di un release di GitHub formattato come ultralytics si aspetta.
        # La via più semplice è se l'utente mette "yolov11n-face.pt" nella stessa directory o in un percorso noto.
        # Per ora, proviamo a caricarlo direttamente, assumendo che sia scaricato o nello CWD/PATH.
        # Se il file non è un formato che YOLO("path/to/model.pt") può caricare direttamente, fallirà.
        # YOLOv11 potrebbe non essere direttamente compatibile con la classe YOLO di ultralytics v8.
        # Prioritizziamo il caricamento locale e poi il fallback.

        # Tentativo 1: Caricare direttamente il file se l'utente l'ha messo nel CWD o in un path rilevabile da YOLO
        try:
            logger.info(f"Attempting to load local '{face_model_filename}'...")
            self.yolo_face_model = YOLO(face_model_filename)
            logger.info(f"Successfully loaded '{face_model_filename}'.")
        except Exception as e_local:
            logger.warning(f"Could not load local '{face_model_filename}': {e_local}. Attempting download if known or fallback.")
            self.yolo_face_model = None # Assicura che sia None se il caricamento locale fallisce

        # Se il caricamento locale fallisce, e se avessimo un meccanismo di download automatico per questo specifico file,
        # lo inseriremmo qui. Dato che attempt_download_asset di ultralytics è per i loro asset,
        # e hf_hub_download era per il modello precedente, per questo URL specifico di GitHub
        # dovremmo implementare un download custom o richiedere all'utente di farlo.
        # Per semplicità, per ora, se il caricamento locale fallisce, passiamo al fallback.

        if self.yolo_face_model is None:
            logger.warning(f"'{face_model_filename}' not found or failed to load. Falling back to MediaPipe for face detection.")
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
