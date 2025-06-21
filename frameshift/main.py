"""Command-line interface for FrameShift."""
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import cv2
import numpy as np
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .utils.detection import Detector
from .utils.crop import union_boxes, smooth_box, compute_crop


def sample_crop(video_path: str, start: int, end: int, detector: Detector,
                frame_w: int, frame_h: int, aspect_ratio: float) -> Tuple[int, int, int, int]:
    """Return crop box computed from a sample of frames between start and end."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    boxes: List[Tuple[int, int, int, int]] = []
    for _ in range(start, min(end, start + 30)):
        ret, frame = cap.read()
        if not ret:
            break
        faces, objs = detector.detect(frame)
        boxes.extend(faces + objs)
    cap.release()
    union = union_boxes(boxes)
    if union is None:
        union = (0, 0, frame_w, frame_h)
    return compute_crop(union, frame_w, frame_h, aspect_ratio)


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


def process_video(
    input_path: str,
    output_path: str,
    ratio: str,
    mode: str = "tracking",
    blur_size: int = 21,
    overlay_opacity: float = 1.0,
) -> None:
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
    pan_end = None
    start = scene_bounds[0][0]
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
            pan_end = None
            if mode == "stationary":
                prev_box = np.array(
                    sample_crop(
                        input_path,
                        frame_idx,
                        current_scene_end,
                        detector,
                        width,
                        height,
                        aspect_ratio,
                    )
                )

        if mode == "stationary":
            crop_box = prev_box
        elif mode == "panning":
            if prev_box is None:
                start_crop = sample_crop(input_path, frame_idx, current_scene_end,
                                         detector, width, height, aspect_ratio)
                end_crop = sample_crop(input_path,
                                       max(frame_idx, current_scene_end - 30),
                                       current_scene_end,
                                       detector, width, height, aspect_ratio)
                prev_box = np.array(start_crop)
                pan_end = np.array(end_crop)
            alpha = (frame_idx - start) / max(1, current_scene_end - start)
            crop_box = (prev_box * (1 - alpha) + pan_end * alpha).astype(int)
        else:  # tracking
            faces, objects = detector.detect(frame)
            all_boxes = faces + objects
            union = union_boxes(all_boxes)
            if union is None:
                union = (0, 0, width, height)
            crop_box = compute_crop(union, width, height, aspect_ratio)
            box_arr = np.array(crop_box)
            crop_box = smooth_box(prev_box, box_arr)
            prev_box = crop_box

        x1, y1, cw, ch = crop_box
        cropped = frame[y1:y1+ch, x1:x1+cw]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if overlay_opacity < 1.0:
            background = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h)), (0, 0), blur_size)
            resized = cv2.addWeighted(resized, overlay_opacity, background, 1 - overlay_opacity, 0)
        out.write(resized)

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameShift auto reframing tool")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--ratio", default="9/16", help="Target aspect ratio (e.g., 9/16)")
    parser.add_argument("--mode", choices=["tracking", "stationary", "panning"], default="tracking",
                        help="Cropping strategy")
    parser.add_argument("--blur", type=int, default=21, help="Blur kernel size for padding")
    parser.add_argument("--overlay", type=float, default=1.0, help="Cropped overlay opacity")
    parser.add_argument("--batch", action="store_true", help="Process all videos in input directory")

    args = parser.parse_args()

    if args.batch:
        in_dir = Path(args.input)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for vid in in_dir.iterdir():
            if vid.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
                continue
            out_path = out_dir / f"{vid.stem}_reframed.mp4"
            process_video(str(vid), str(out_path), args.ratio, args.mode, args.blur, args.overlay)
    else:
        process_video(args.input, args.output, args.ratio, args.mode, args.blur, args.overlay)


if __name__ == "__main__":
    main()
