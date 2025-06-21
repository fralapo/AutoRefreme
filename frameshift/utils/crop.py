"""Functions for crop box computation and smoothing."""
from typing import List, Tuple, Optional
import numpy as np


def union_boxes(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def smooth_box(prev_box: Optional[np.ndarray], curr_box: np.ndarray, factor: float = 0.2) -> np.ndarray:
    if prev_box is None:
        return curr_box
    return (prev_box * (1 - factor) + curr_box * factor).astype(int)


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
