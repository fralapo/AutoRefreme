"""Command-line interface for FrameShift."""
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional # Added Optional
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque # Added deque
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .utils.detection import Detector
from .utils.crop import union_boxes, compute_crop, calculate_weighted_interest_region
from .weights_parser import parse_object_weights


def sample_crop(video_path: str, start_frame_num: int, end_frame_num: int, detector: Detector,
                frame_w: int, frame_h: int, aspect_ratio: float,
                object_weights_map: Dict[str, float]) -> Tuple[int, int, int, int]: # Added object_weights_map
    """Return crop box computed from a sample of frames between start_frame_num and end_frame_num."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return compute_crop((0,0,frame_w,frame_h), frame_w, frame_h, aspect_ratio)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)

    # Accumulate all detections from the sample window
    all_detections_in_sample: List[Dict[str, Any]] = []

    current_processed_frame_count = 0
    MAX_FRAMES_TO_SAMPLE_INTERNAL = 150

    for i in range(start_frame_num, end_frame_num):
        if current_processed_frame_count >= MAX_FRAMES_TO_SAMPLE_INTERNAL:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # detector.detect() now returns List[Dict[str, Any]]
        faces_detected, objects_detected = detector.detect(frame)
        # We need to associate these detections with the specific frame if we do temporal analysis later,
        # but for now, just collect all detections in the window.
        all_detections_in_sample.extend(faces_detected)
        all_detections_in_sample.extend(objects_detected)
        current_processed_frame_count += 1

    cap.release()

    # Calculate a single interest region guide box based on *all* detections in the sample window
    # This is a key change: instead of averaging boxes, we find a weighted interest region
    # from the aggregate of detections over the sample period.
    interest_guide_box = calculate_weighted_interest_region(
        all_detections_in_sample, object_weights_map, frame_w, frame_h
    )

    return compute_crop(interest_guide_box, frame_w, frame_h, aspect_ratio)


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
    padding_style: str = "fill",
    blur_amount: int = 21,
    content_opacity: float = 1.0,
    object_weights_map: Dict[str, float] = None,
) -> None:
    if object_weights_map is None:
        object_weights_map = {'face': 1.0, 'person': 0.8, 'default': 0.5}
    # La modalità è implicitamente 'stationary'

    # Helper function for padding styles 'blur' and 'black'
    # This could also be moved to utils/crop.py if preferred
    def _apply_fit_padding(base_frame_template: np.ndarray, content_to_fit: np.ndarray,
                           target_w: int, target_h: int, is_blur_bg: bool = False, blur_kernel_size: int = 21, original_frame_for_blur = None) -> np.ndarray:

        output_frame = base_frame_template.copy() # Start with black or pre-blurred background

        if content_to_fit.shape[0] == 0 or content_to_fit.shape[1] == 0:
            return output_frame

        content_h, content_w = content_to_fit.shape[:2]
        if content_h == 0 or content_w == 0:
            return output_frame

        scale_h = target_h / content_h
        scale_w = target_w / content_w
        scale = min(scale_h, scale_w)

        scaled_w = int(content_w * scale)
        scaled_h = int(content_h * scale)

        if scaled_w == 0 or scaled_h == 0:
            return output_frame

        resized_content = cv2.resize(content_to_fit, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        pad_x = (target_w - scaled_w) // 2
        pad_y = (target_h - scaled_h) // 2

        dst_x_start = max(0, pad_x)
        dst_y_start = max(0, pad_y)

        copy_w = min(scaled_w, target_w - dst_x_start)
        copy_h = min(scaled_h, target_h - dst_y_start)

        src_x_start = 0
        src_y_start = 0

        if copy_w > 0 and copy_h > 0:
            # Ensure the slice from resized_content is valid
            src_slice_w = min(copy_w, resized_content.shape[1])
            src_slice_h = min(copy_h, resized_content.shape[0])

            # Ensure the destination slice is valid
            dst_slice_w = min(copy_w, output_frame.shape[1] - dst_x_start)
            dst_slice_h = min(copy_h, output_frame.shape[0] - dst_y_start)

            # Final actual copy dimensions must match
            final_copy_w = min(src_slice_w, dst_slice_w)
            final_copy_h = min(src_slice_h, dst_slice_h)

            if final_copy_w > 0 and final_copy_h > 0:
                # Resize content to fit the exact calculated copy dimensions before placing
                content_final_size = cv2.resize(resized_content, (final_copy_w, final_copy_h), interpolation=cv2.INTER_AREA)
                output_frame[dst_y_start : dst_y_start + final_copy_h,
                             dst_x_start : dst_x_start + final_copy_w] = content_final_size
        return output_frame

    if ":" in ratio:
        w_str, h_str = ratio.split(':')
        aspect_ratio = float(w_str) / float(h_str)
    else:
        aspect_ratio = float(ratio)
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
    # prev_box è usato dalla modalità stationary per memorizzare il box fisso della scena.
    prev_box: Optional[np.ndarray] = None

    start = scene_bounds[0][0] # Start frame of the current scene
    current_scene_end = scene_bounds[0][1] # End frame of the current scene
    scene_iter = iter(scene_bounds)
    _ = next(scene_iter) # Consuma la prima scena dall'iteratore

    # Inizializzazione di prev_box per la prima scena (modalità stationary implicita)
    # print(f"DEBUG: Initializing prev_box for stationary mode for first scene ({start}-{current_scene_end})")
    initial_crop_for_stationary = sample_crop(
        input_path,
        start,
        current_scene_end,
        detector,
        width,
        height,
        aspect_ratio,
        object_weights_map, # Passare i pesi
    )
    prev_box = np.array(initial_crop_for_stationary)
    # print(f"DEBUG: Initial prev_box for stationary: {prev_box}")

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

            # Ricalcola il box fisso per la nuova scena (modalità stationary implicita)
            prev_box = np.array(
                sample_crop(
                    input_path,
                    start,
                    current_scene_end,
                    detector,
                    width,
                    height,
                    aspect_ratio,
                    object_weights_map,
                )
            )

        # Determinazione del crop_box per il frame corrente (sempre stationary)
        if prev_box is None: # Fallback di sicurezza, non dovrebbe accadere con la logica sopra
            # print("ERROR: prev_box is None in stationary mode unexpectedly.")
            crop_box = np.array([0,0,width,height]) # Fallback a full frame
        else:
            crop_box = prev_box


        # Fallback se crop_box è None (non dovrebbe accadere con le correzioni precedenti, ma per sicurezza)
        if crop_box is None:
            # print(f"WARNING: crop_box is None at frame {frame_idx} before unpacking. Mode: {mode}. Defaulting.")
            if prev_box is not None and isinstance(prev_box, np.ndarray) and prev_box.size == 4:
                # Usa l'ultimo prev_box valido se è un ndarray (formato atteso x,y,w,h)
                # print(f"DEBUG: Fallback to prev_box: {prev_box}")
                x1_crop, y1_crop, cw_crop, ch_crop = prev_box.astype(int)
            else:
                # Fallback estremo all'intero frame
                # print(f"DEBUG: Fallback to full frame crop (0,0,width,height).")
                x1_crop, y1_crop, cw_crop, ch_crop = 0, 0, width, height
        elif isinstance(crop_box, np.ndarray):
            x1_crop, y1_crop, cw_crop, ch_crop = crop_box.astype(int)
        else: # Dovrebbe essere una tupla o lista se non ndarray e non None
            x1_crop, y1_crop, cw_crop, ch_crop = map(int, crop_box)

        cropped_content = frame[y1_crop:y1_crop+ch_crop, x1_crop:x1_crop+cw_crop]

        # final_frame_output sarà determinato dal padding_style

        if padding_style == 'blur':
            # Crea sfondo sfocato
            background_for_blur = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA), (0,0), blur_amount)
            final_frame_output = _apply_fit_padding(background_for_blur, cropped_content, out_w, out_h)

        elif padding_style == 'black':
            background_black = np.zeros((out_h, out_w, 3), dtype=np.uint8) # Sfondo nero
            final_frame_output = _apply_fit_padding(background_black, cropped_content, out_w, out_h)

        elif padding_style == 'fill': # Default: Fill / Pan & Scan
            # Logica per FILL / PAN & SCAN
            final_frame_output = np.zeros((out_h, out_w, 3), dtype=np.uint8) # Inizia nero, ma verrà riempito
            if cropped_content.shape[0] > 0 and cropped_content.shape[1] > 0:
                content_h, content_w = cropped_content.shape[:2]
                if content_h > 0 and content_w > 0:
                    scale_h_fill = out_h / content_h
                    scale_w_fill = out_w / content_w
                    scale_fill = max(scale_h_fill, scale_w_fill)

                    scaled_content_fill_w = int(content_w * scale_fill)
                    scaled_content_fill_h = int(content_h * scale_fill)

                    if scaled_content_fill_w > 0 and scaled_content_fill_h > 0:
                        content_to_process = cv2.resize(cropped_content, (scaled_content_fill_w, scaled_content_fill_h), interpolation=cv2.INTER_AREA)

                        src_x = (scaled_content_fill_w - out_w) // 2
                        src_y = (scaled_content_fill_h - out_h) // 2
                        dst_x = 0; dst_y = 0
                        copy_width = out_w; copy_height = out_h

                        if scaled_content_fill_w < out_w:
                            dst_x = (out_w - scaled_content_fill_w) // 2; src_x = 0; copy_width = scaled_content_fill_w
                        if scaled_content_fill_h < out_h:
                            dst_y = (out_h - scaled_content_fill_h) // 2; src_y = 0; copy_height = scaled_content_fill_h

                        src_x = max(0, src_x); src_y = max(0, src_y)

                        actual_copy_w = min(copy_width, content_to_process.shape[1] - src_x, out_w - dst_x)
                        actual_copy_h = min(copy_height, content_to_process.shape[0] - src_y, out_h - dst_y)

                        if actual_copy_w > 0 and actual_copy_h > 0:
                            src_slice = content_to_process[src_y : src_y + actual_copy_h, src_x : src_x + actual_copy_w]
                            if src_slice.shape[0] == actual_copy_h and src_slice.shape[1] == actual_copy_w:
                                final_frame_output[dst_y : dst_y + actual_copy_h, dst_x : dst_x + actual_copy_w] = src_slice
            # Se cropped_content era vuoto o le sue dimensioni scalate non valide, final_frame_output rimane nero.
        else: # Stile di padding non riconosciuto, fallback a nero (o fill)
            print(f"Warning: Unknown padding_style '{padding_style}'. Defaulting to 'fill'.")
            # Copia-incolla della logica 'fill' per sicurezza come fallback
            final_frame_output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            if cropped_content.shape[0] > 0 and cropped_content.shape[1] > 0:
                content_h, content_w = cropped_content.shape[:2]
                if content_h > 0 and content_w > 0:
                    scale_h_fill = out_h / content_h; scale_w_fill = out_w / content_w
                    scale_fill = max(scale_h_fill, scale_w_fill)
                    scaled_content_fill_w = int(content_w * scale_fill); scaled_content_fill_h = int(content_h * scale_fill)
                    if scaled_content_fill_w > 0 and scaled_content_fill_h > 0:
                        content_to_process = cv2.resize(cropped_content, (scaled_content_fill_w, scaled_content_fill_h), interpolation=cv2.INTER_AREA)
                        src_x = (scaled_content_fill_w - out_w) // 2; src_y = (scaled_content_fill_h - out_h) // 2
                        dst_x = 0; dst_y = 0
                        copy_width = out_w; copy_height = out_h
                        if scaled_content_fill_w < out_w: dst_x = (out_w - scaled_content_fill_w) // 2; src_x = 0; copy_width = scaled_content_fill_w
                        if scaled_content_fill_h < out_h: dst_y = (out_h - scaled_content_fill_h) // 2; src_y = 0; copy_height = scaled_content_fill_h
                        src_x = max(0, src_x); src_y = max(0, src_y)
                        actual_copy_w = min(copy_width, content_to_process.shape[1] - src_x, out_w - dst_x)
                        actual_copy_h = min(copy_height, content_to_process.shape[0] - src_y, out_h - dst_y)
                        if actual_copy_w > 0 and actual_copy_h > 0:
                            src_slice = content_to_process[src_y : src_y + actual_copy_h, src_x : src_x + actual_copy_w]
                            if src_slice.shape[0] == actual_copy_h and src_slice.shape[1] == actual_copy_w:
                                final_frame_output[dst_y : dst_y + actual_copy_h, dst_x : dst_x + actual_copy_w] = src_slice

        # Applica l'opacità generale se content_opacity < 1.0
        # Questa logica si applica dopo che final_frame_output è stato costruito (con contenuto e padding/fill)
        if content_opacity < 1.0:
            # Lo sfondo per l'opacità è sempre il frame originale sfocato e ridimensionato a output
            full_frame_blur_bg_for_opacity = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA), (0,0), blur_amount)
            final_frame_output = cv2.addWeighted(final_frame_output, content_opacity, full_frame_blur_bg_for_opacity, 1 - content_opacity, 0)

        out.write(final_frame_output)

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameShift auto reframing tool. Default mode is 'stationary'.")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--ratio", default="9/16", help="Target aspect ratio (e.g., 9/16)")
    parser.add_argument(
        "--padding_style",
        type=str,
        default="fill",
        choices=['fill', 'black', 'blur'],
        help="Padding style. 'fill': Pan & Scan, content fills frame (default). "
             "'black': Fit content, use black bars. "
             "'blur': Fit content, use blurred background bars."
    )
    parser.add_argument("--blur_amount", type=int, default=21, help="Blur kernel size for 'blur' padding_style.")
    parser.add_argument("--content_opacity", type=float, default=1.0, help="Opacity of the main content. If < 1.0, content is blended with the background.")
    # Nota: l'argomento --enable_padding è stato rimosso. Usare --padding_style.
    parser.add_argument(
        "--object_weights",
        type=str,
        default="face:1.0,person:0.8,default:0.5", # Default weights
        help="Comma-separated string of 'label:weight' pairs (e.g., \"face:1.0,person:0.8,default:0.5\"). "
             "Assigns importance weights to detected object classes for stationary mode. 'default' is used for unspecified classes."
    )
    # Smoothing/deadzone arguments are removed as they were tracking specific
    parser.add_argument("--batch", action="store_true", help="Process all videos in input directory")

    args = parser.parse_args()

    if args.batch:
        in_dir = Path(args.input)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        parsed_weights = parse_object_weights(args.object_weights)

        for vid in in_dir.iterdir():
            if vid.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
                continue
            out_path = out_dir / f"{vid.stem}_reframed.mp4"
            process_video(str(vid), str(out_path), args.ratio,
                          padding_style=args.padding_style, blur_amount=args.blur_amount,
                          content_opacity=args.content_opacity, object_weights_map=parsed_weights)
    else:
        parsed_weights = parse_object_weights(args.object_weights)
        process_video(args.input, args.output, args.ratio,
                      padding_style=args.padding_style, blur_amount=args.blur_amount,
                      content_opacity=args.content_opacity, object_weights_map=parsed_weights)


if __name__ == "__main__":
    main()
