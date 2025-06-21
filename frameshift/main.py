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
    enable_padding: bool = False,
    blur_amount: int = 21,
    content_opacity: float = 1.0,
) -> None:
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
    prev_box = None
    pan_end = None
    start = scene_bounds[0][0]
    current_scene_end = scene_bounds[0][1]
    scene_iter = iter(scene_bounds)
    _ = next(scene_iter) # Consuma la prima scena dall'iteratore, 'start' e 'current_scene_end' si riferiscono alla prima scena

    # Inizializzazione di prev_box per la prima scena se in modalità stazionaria
    if mode == "stationary":
        # print(f"DEBUG: Initializing prev_box for stationary mode for first scene ({start}-{current_scene_end})")
        initial_crop_for_stationary = sample_crop(
            input_path,
            start,
            current_scene_end,
            detector,
            width,
            height,
            aspect_ratio,
        )
        # sample_crop dovrebbe sempre restituire una tupla valida (x,y,w,h)
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

        final_frame_output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        if enable_padding:
            blurred_background = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA), (0,0), blur_amount)

            content_h, content_w = cropped_content.shape[:2]
            if content_h == 0 or content_w == 0:
                # Se il crop è vuoto, il contenuto ridimensionato riempirà l'output come nero (o sfondo sfocato se opacità < 1)
                resized_content_for_padding = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                final_content_w, final_content_h = out_w, out_h
            else:
                scale_h = out_h / content_h
                scale_w = out_w / content_w
                scale = min(scale_h, scale_w)
                final_content_w = int(content_w * scale)
                final_content_h = int(content_h * scale)
                resized_content_for_padding = cv2.resize(cropped_content, (final_content_w, final_content_h), interpolation=cv2.INTER_AREA)

            pad_x = (out_w - final_content_w) // 2
            pad_y = (out_h - final_content_h) // 2

            final_frame_output = blurred_background.copy()

            # Prepara la porzione di sfondo su cui il contenuto verrà sovrapposto
            # Assicurati che le fette non siano negative o fuori dai limiti se final_content_w/h sono più grandi di out_w/h (non dovrebbe succedere con min(scale))
            bg_slice_y_start = max(0, pad_y)
            bg_slice_y_end = min(out_h, pad_y + final_content_h)
            bg_slice_x_start = max(0, pad_x)
            bg_slice_x_end = min(out_w, pad_x + final_content_w)

            content_slice_h = bg_slice_y_end - bg_slice_y_start
            content_slice_w = bg_slice_x_end - bg_slice_x_start

            if content_slice_h > 0 and content_slice_w > 0: # Solo se l'area di sovrapposizione è valida
                # Estrai la porzione di sfondo corrispondente al contenuto ridimensionato
                center_of_background = blurred_background[bg_slice_y_start:bg_slice_y_end, bg_slice_x_start:bg_slice_x_end]

                # Adatta resized_content_for_padding se le dimensioni della fetta sono diverse (ad es. a causa di arrotondamenti/limiti)
                actual_content_to_blend = cv2.resize(resized_content_for_padding, (content_slice_w, content_slice_h), interpolation=cv2.INTER_AREA)

                if content_opacity < 1.0:
                    if center_of_background.shape[:2] == actual_content_to_blend.shape[:2]:
                        blended_content = cv2.addWeighted(actual_content_to_blend, content_opacity, center_of_background, 1 - content_opacity, 0)
                        final_frame_output[bg_slice_y_start:bg_slice_y_end, bg_slice_x_start:bg_slice_x_end] = blended_content
                    else: # Fallback se le dimensioni non corrispondono (non dovrebbe accadere con la logica sopra)
                        final_frame_output[bg_slice_y_start:bg_slice_y_end, bg_slice_x_start:bg_slice_x_end] = actual_content_to_blend
                else:
                    final_frame_output[bg_slice_y_start:bg_slice_y_end, bg_slice_x_start:bg_slice_x_end] = actual_content_to_blend
            # Se content_slice_h/w è 0, final_frame_output rimane lo sfondo sfocato (corretto)

        else: # Comportamento senza padding esplicito (vecchia logica adattata)
            if cropped_content.shape[0] == 0 or cropped_content.shape[1] == 0:
                 direct_resized_content = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            else:
                direct_resized_content = cv2.resize(cropped_content, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if content_opacity < 1.0:
                full_frame_blur_bg = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA), (0,0), blur_amount)
                final_frame_output = cv2.addWeighted(direct_resized_content, content_opacity, full_frame_blur_bg, 1 - content_opacity, 0)
            else:
                final_frame_output = direct_resized_content

        out.write(final_frame_output)

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
    parser.add_argument("--enable_padding", action="store_true", help="Enable letterbox/pillarbox padding with blurred background. Uses --blur_amount and --content_opacity.")
    parser.add_argument("--blur_amount", type=int, default=21, help="Blur kernel size for padding background or full overlay background.")
    parser.add_argument("--content_opacity", type=float, default=1.0, help="Opacity of the main content. If < 1.0, content is blended with the background (either bars or full frame blur).")
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
            process_video(str(vid), str(out_path), args.ratio, args.mode, args.enable_padding, args.blur_amount, args.content_opacity)
    else:
        process_video(args.input, args.output, args.ratio, args.mode, args.enable_padding, args.blur_amount, args.content_opacity)


if __name__ == "__main__":
    main()
