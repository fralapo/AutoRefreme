"""Functions for crop box computation and smoothing."""
from typing import List, Tuple, Optional, Dict, Any # Added Dict, Any
import numpy as np
from collections import deque


def union_boxes(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def smooth_box(prev_box: Optional[np.ndarray], curr_box: np.ndarray, factor: float = 0.2) -> np.ndarray:
    if prev_box is None:
        return curr_box
    return (prev_box * (1 - factor) + curr_box * factor).astype(int)

# Renaming old smooth_box
smooth_box_legacy = smooth_box

def smooth_box_windowed(
    historical_raw_boxes: deque, # Deque of np.ndarray [(x,y,w,h), ...], does NOT include current_raw_box
    current_raw_box: np.ndarray, # np.ndarray [x,y,w,h] - current un-smoothed box from compute_crop
    responsiveness: float,       # How much weight to give the current_raw_box (0.0 to 1.0)
    prev_smoothed_box: Optional[np.ndarray] # Smoothed box from the previous frame
) -> np.ndarray:
    """
    Smooths the current_raw_box based on a historical window of raw boxes and the previous smoothed box.
    Boxes are in (x, y, width, height) format.
    """
    # If no history and no previous smoothed box (very first frame of tracking), return current raw box.
    if not historical_raw_boxes and prev_smoothed_box is None:
        return current_raw_box

    # If history is empty but there's a prev_smoothed_box (e.g., first few frames of tracking),
    # use legacy smoothing logic between prev_smoothed and current_raw.
    if not historical_raw_boxes:
        return (prev_smoothed_box * (1 - responsiveness) + current_raw_box * responsiveness).astype(int)

    # Calculate the mean of the boxes in the historical window
    # These are raw (un-smoothed) boxes from previous frames.
    historical_mean_box = np.mean(list(historical_raw_boxes), axis=0)

    # Interpolate between the historical mean and the current raw box
    smoothed_box = (historical_mean_box * (1 - responsiveness) + current_raw_box * responsiveness).astype(int)

    return smoothed_box


def compute_crop(box: Tuple[int, int, int, int], frame_w: int, frame_h: int, aspect_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w // 2
    cy = y1 + box_h // 2

    crop_h = max(box_h, int(box_w / aspect_ratio))
    crop_w = int(crop_h * aspect_ratio)
    if crop_w < box_w:
        crop_w = box_w
        crop_h = int(crop_w / aspect_ratio)

    x1 = max(0, min(frame_w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(frame_h - crop_h, cy - crop_h // 2))
    return x1, y1, crop_w, crop_h


def calculate_weighted_interest_region(
    detections: List[Dict[str, Any]],
    object_weights: Dict[str, float],
    frame_w: int,
    frame_h: int
) -> Tuple[int, int, int, int]:
    """
    Calculates a bounding box representing the weighted area of interest.
    Returns a box (x1, y1, x2, y2) to be used by compute_crop.
    The returned box will have dimensions equal to the union of all detections,
    but its center will be the weighted centroid of all detections.
    """
    if not detections:
        return (0, 0, frame_w, frame_h)

    total_effective_weight = 0.0
    weighted_cx_sum = 0.0
    weighted_cy_sum = 0.0

    all_boxes_for_union: List[Tuple[int, int, int, int]] = []

    for det in detections:
        box = det['box'] # (x1, y1, x2, y2)
        label = det['label'].lower()
        # confidence = det['confidence'] # Potremmo usarla in futuro

        all_boxes_for_union.append(box)

        weight = object_weights.get(label, object_weights.get('default', 0.5))
        if weight == 0:
            continue

        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h
        if area == 0: # Evita di usare box senza area se non hanno peso
            if weight > 0 : # Se ha peso ma area 0, considera un'area minima per dargli un contributo al centro
                area = 1
            else: # Peso 0 e area 0, ignora
                continue


        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Ponderazione: diamo più importanza ai box più grandi e a quelli con etichetta di peso elevato.
        # Usare l'area come parte del peso effettivo può aiutare a stabilizzare
        # il centro se ci sono molti piccoli oggetti vs uno grande.
        # Potremmo anche usare sqrt(area) o solo il peso dell'etichetta.
        # Iniziamo con peso_etichetta * (1 + sqrt(area)) per dare un contributo base anche a oggetti piccoli ma importanti
        effective_weight = weight * (1 + np.sqrt(area / max(1, frame_w * frame_h)) * 10) # Normalizza area e scala
        # effective_weight = weight # Alternativa più semplice

        weighted_cx_sum += cx * effective_weight
        weighted_cy_sum += cy * effective_weight
        total_effective_weight += effective_weight

    if total_effective_weight == 0: # Nessun oggetto con peso > 0 o area > 0
        # Fallback: usa il centro del semplice union_boxes se non ci sono pesi effettivi
        # o se tutti i pesi erano 0.
        # Tuttavia, union_boxes potrebbe essere None se all_boxes_for_union è vuoto (non dovrebbe succedere se detections non è vuoto)
        # Ma se tutti i detection avessero peso 0, all_boxes_for_union sarebbe popolato.
        union_of_detections = union_boxes(all_boxes_for_union)
        if union_of_detections is None: # Dovrebbe accadere solo se detections era vuoto, già gestito.
             return (0,0, frame_w, frame_h)
        # print("DEBUG: No effective weights, using geometric center of union_boxes.")
        ux1, uy1, ux2, uy2 = union_of_detections
        final_cx = (ux1 + ux2) / 2.0
        final_cy = (uy1 + uy2) / 2.0
        union_w = ux2 - ux1
        union_h = uy2 - uy1
    else:
        final_cx = weighted_cx_sum / total_effective_weight
        final_cy = weighted_cy_sum / total_effective_weight

        # Ora calcoliamo le dimensioni del box unione
        union_of_detections = union_boxes(all_boxes_for_union)
        if union_of_detections is None: # Non dovrebbe accadere se detections non è vuoto
            return (0,0, frame_w, frame_h)
        ux1, uy1, ux2, uy2 = union_of_detections
        union_w = ux2 - ux1
        union_h = uy2 - uy1


    # Costruisci il "guide box" per compute_crop: dimensioni dell'unione, centrato sul baricentro pesato.
    guide_x1 = int(final_cx - union_w / 2.0)
    guide_y1 = int(final_cy - union_h / 2.0)
    guide_x2 = int(final_cx + union_w / 2.0)
    guide_y2 = int(final_cy + union_h / 2.0)

    # Assicura che il box guida sia all'interno dei limiti del frame, anche se compute_crop lo farebbe.
    # Questo è più per la validità del box (x1<x2).
    # compute_crop gestisce il clipping finale del crop_box risultante.
    # Non è strettamente necessario clippare qui il guide_box, ma può evitare guide_box non validi.
    # guide_x1 = max(0, guide_x1)
    # guide_y1 = max(0, guide_y1)
    # guide_x2 = min(frame_w, guide_x2)
    # guide_y2 = min(frame_h, guide_y2)
    # if guide_x1 >= guide_x2 or guide_y1 >= guide_y2: # Se il box è invalido
    #     return ux1, uy1, ux2, uy2 # Fallback al semplice union box

    # print(f"DEBUG: Weighted Region: Original Union: {(ux1,uy1,ux2,uy2)}, Guide Box: {(guide_x1, guide_y1, guide_x2, guide_y2)}")
    return (guide_x1, guide_y1, guide_x2, guide_y2)
