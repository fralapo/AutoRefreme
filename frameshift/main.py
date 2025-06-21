"""Command-line interface for FrameShift."""
import argparse
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .utils.detection import Detector
from .utils.crop import union_boxes, smooth_box, compute_crop


def detect_scenes(video_path: str) -> Dict[int, int]:
    """Return a dictionary mapping start_frame to end_frame for each scene."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    if not scenes:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {0: total}
    return {s[0].get_frames(): s[1].get_frames() for s in scenes}


def process_video(input_path: str, output_path: str, ratio: str) -> None:
    aspect_ratio = eval(ratio) if ":" in ratio else float(ratio)
    detector = Detector()
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scenes = detect_scenes(input_path)
    scene_bounds = sorted(scenes.items())

    out_w = int(height * aspect_ratio)
    out_h = height
    if out_w > width:
        out_w = width
        out_h = int(out_w / aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    prev_box = None
    current_scene_end = scene_bounds[0][1]
    scene_iter = iter(scene_bounds)
    _ = next(scene_iter)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= current_scene_end:
            try:
                start, current_scene_end = next(scene_iter)
            except StopIteration:
                current_scene_end = float('inf')
            prev_box = None

        faces, objects = detector.detect(frame)
        all_boxes = faces + objects
        union = union_boxes(all_boxes)
        if union is None:
            union = (0, 0, width, height)
        crop_box = compute_crop(union, width, height, aspect_ratio)
        box_arr = np.array(crop_box)
        smooth = smooth_box(prev_box, box_arr)
        prev_box = smooth

        x1, y1, cw, ch = smooth
        cropped = frame[y1:y1+ch, x1:x1+cw]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out.write(resized)

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameShift auto reframing tool")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--ratio", default="9/16", help="Target aspect ratio (e.g., 9/16)")
    args = parser.parse_args()
    process_video(args.input, args.output, args.ratio)


if __name__ == "__main__":
    main()
