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
# Renamed smooth_box to smooth_box_legacy, new one is smooth_box_windowed
from .utils.crop import union_boxes, smooth_box_legacy, smooth_box_windowed, compute_crop, calculate_weighted_interest_region
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
    mode: str = "tracking",
    enable_padding: bool = False,
    blur_amount: int = 21,
    content_opacity: float = 1.0,
    tracking_responsiveness: float = 0.2,
    object_weights_map: Dict[str, float] = None, # Will be populated by parsed arg
    smoothing_window_size: int = 5,
    tracking_deadzone_center_px: int = 10,
    tracking_deadzone_size_percent: float = 0.05,
) -> None:
    if object_weights_map is None: # Default if not passed (e.g. direct call)
        object_weights_map = {'face': 1.0, 'person': 0.8, 'default': 0.5}

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
    # prev_box is now prev_smoothed_crop_box, storing the output of the smoothing function
    prev_smoothed_crop_box: Optional[np.ndarray] = None
    # historical_raw_boxes stores raw (un-smoothed) boxes from compute_crop for windowed smoothing
    historical_raw_boxes: deque = deque(maxlen=max(1, smoothing_window_size)) # maxlen must be > 0

    # prev_box is used by stationary and panning modes to store scene-specific state
    # For stationary, it's the fixed box for the scene.
    # For panning, it's the starting box for the pan.
    prev_box: Optional[np.ndarray] = None
    pan_end: Optional[np.ndarray] = None # Specific to panning mode

    start = scene_bounds[0][0] # Start frame of the current scene
    current_scene_end = scene_bounds[0][1] # End frame of the current scene
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
            object_weights_map, # Passare i pesi
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

            # Reset per stati dipendenti dalla scena
            pan_end = None # Per panning
            # 'prev_box' è usato da stationary e panning (come start_box)
            # Viene ricalcolato sotto se mode == stationary o se mode == panning e prev_box è None
            prev_box = None

            if mode == "tracking":
                historical_raw_boxes.clear()
                prev_smoothed_crop_box = None

            if mode == "stationary":
                # Ricalcola il box fisso per la nuova scena usando 'start' (inizio della nuova scena)
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

        # Determinazione del crop_box per il frame corrente
        if mode == "stationary":
            # prev_box è stato calcolato all'inizio di process_video o al cambio di scena.
            # Deve essere un np.ndarray valido qui.
            if prev_box is None: # Fallback di sicurezza, non dovrebbe accadere
                # print("ERROR: prev_box is None in stationary mode unexpectedly.")
                crop_box = np.array([0,0,width,height]) # Fallback a full frame
            else:
                crop_box = prev_box
        elif mode == "panning":
            # prev_box qui funge da pan_start_box.
            if prev_box is None:
                scene_duration = current_scene_end - start
                sample_window_len = min(max(30, scene_duration // 4), 150)

                start_crop_sample_start_ts = start
                start_crop_sample_end_ts = min(current_scene_end, start + sample_window_len)

                end_crop_sample_start_ts = max(start, current_scene_end - sample_window_len)
                end_crop_sample_end_ts = current_scene_end

                if scene_duration < 2 * sample_window_len and scene_duration > 0 : # Scena corta, evita sovrapposizione eccessiva
                    mid_point = start + scene_duration // 2
                    start_crop_sample_end_ts = mid_point
                    end_crop_sample_start_ts = mid_point

                # Assicura che le finestre di campionamento siano valide (start < end)
                if start_crop_sample_start_ts >= start_crop_sample_end_ts:
                    start_crop_sample_end_ts = min(current_scene_end, start_crop_sample_start_ts + 1) if current_scene_end > start_crop_sample_start_ts else start_crop_sample_start_ts

                if end_crop_sample_start_ts >= end_crop_sample_end_ts:
                    end_crop_sample_start_ts = max(start, end_crop_sample_end_ts - 1) if start < end_crop_sample_end_ts else end_crop_sample_end_ts

                # print(f"DEBUG Panning: Scene: {start}-{current_scene_end} (dur: {scene_duration}), frame_idx: {frame_idx}")
                # print(f"DEBUG Panning: Start crop sample window: [{start_crop_sample_start_ts}, {start_crop_sample_end_ts})")
                # print(f"DEBUG Panning: End crop sample window:   [{end_crop_sample_start_ts}, {end_crop_sample_end_ts})")

                start_crop = sample_crop(input_path, start_crop_sample_start_ts, start_crop_sample_end_ts,
                                         detector, width, height, aspect_ratio, object_weights_map) # Passare i pesi
                end_crop = sample_crop(input_path, end_crop_sample_start_ts, end_crop_sample_end_ts,
                                       detector, width, height, aspect_ratio, object_weights_map) # Passare i pesi
                prev_box = np.array(start_crop)
                pan_end = np.array(end_crop)

            alpha = (frame_idx - start) / max(1, current_scene_end - start) if current_scene_end > start else 0
            crop_box = (prev_box * (1 - alpha) + pan_end * alpha).astype(int)
        elif mode == "tracking":
            faces_detected, objects_detected = detector.detect(frame)
            all_detections = faces_detected + objects_detected

            interest_guide_box = calculate_weighted_interest_region(
                all_detections, object_weights_map, width, height
            )

            current_raw_crop_tuple = compute_crop(interest_guide_box, width, height, aspect_ratio)
            current_raw_crop_arr = np.array(current_raw_crop_tuple) # (x,y,w,h)

            should_move = True
            if prev_smoothed_crop_box is not None:
                prev_x, prev_y, prev_w, prev_h = prev_smoothed_crop_box
                prev_cx = prev_x + prev_w / 2.0
                prev_cy = prev_y + prev_h / 2.0

                curr_x, curr_y, curr_w, curr_h = current_raw_crop_arr
                curr_cx = curr_x + curr_w / 2.0
                curr_cy = curr_y + curr_h / 2.0

                center_diff = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)

                size_diff_w = abs(curr_w - prev_w) / max(1, float(prev_w)) if prev_w > 0 else (0 if curr_w == 0 else 1.0)
                size_diff_h = abs(curr_h - prev_h) / max(1, float(prev_h)) if prev_h > 0 else (0 if curr_h == 0 else 1.0)

                if (center_diff < tracking_deadzone_center_px and
                    size_diff_w < tracking_deadzone_size_percent and
                    size_diff_h < tracking_deadzone_size_percent):
                    should_move = False

            if should_move:
                smoothed_box_arr = smooth_box_windowed(
                    historical_raw_boxes,
                    current_raw_crop_arr,
                    tracking_responsiveness,
                    prev_smoothed_crop_box
                )
                prev_smoothed_crop_box = smoothed_box_arr # Aggiorna il box smussato per il prossimo frame
                crop_box = smoothed_box_arr
            else: # Non muovere la camera
                crop_box = prev_smoothed_crop_box # Usa l'ultimo box smussato
                # prev_smoothed_crop_box rimane lo stesso, la camera non si è mossa

            # Aggiorna sempre lo storico dei box grezzi
            historical_raw_boxes.append(current_raw_crop_arr)

        else: # Should not happen (se mode non è stationary, panning, o tracking)
            crop_box = np.array([0,0,width,height])


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

        else: # Comportamento senza --enable_padding (ritaglio con barre nere se necessario, no deformazione)
            final_frame_output = np.zeros((out_h, out_w, 3), dtype=np.uint8) # Sfondo nero

            if cropped_content.shape[0] > 0 and cropped_content.shape[1] > 0:
                content_h, content_w = cropped_content.shape[:2]

                scale_h = out_h / content_h
                scale_w = out_w / content_w
                scale = min(scale_h, scale_w)

                scaled_content_w = int(content_w * scale)
                scaled_content_h = int(content_h * scale)

                resized_content = cv2.resize(cropped_content, (scaled_content_w, scaled_content_h), interpolation=cv2.INTER_AREA)

                pad_x = (out_w - scaled_content_w) // 2
                pad_y = (out_h - scaled_content_h) // 2

                slice_y_start = max(0, pad_y)
                slice_y_end = min(out_h, pad_y + scaled_content_h)
                slice_x_start = max(0, pad_x)
                slice_x_end = min(out_w, pad_x + scaled_content_w)

                # Adatta resized_content alle dimensioni effettive della fetta di destinazione
                # Questo è importante se scaled_content_w/h è leggermente diverso da (slice_x_end - slice_x_start) a causa di arrotondamenti
                actual_content_to_place_w = slice_x_end - slice_x_start
                actual_content_to_place_h = slice_y_end - slice_y_start

                if actual_content_to_place_w > 0 and actual_content_to_place_h > 0:
                    actual_content_to_place = cv2.resize(resized_content, (actual_content_to_place_w, actual_content_to_place_h), interpolation=cv2.INTER_AREA)
                    final_frame_output[slice_y_start:slice_y_end, slice_x_start:slice_x_end] = actual_content_to_place

            # Applica l'opacità generale se content_opacity < 1.0
            # Il final_frame_output (contenuto su barre nere, o solo barre nere se cropped_content era vuoto)
            # viene mescolato con uno sfondo sfocato dell'intero frame originale.
            if content_opacity < 1.0:
                full_frame_blur_bg = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA), (0,0), blur_amount)
                final_frame_output = cv2.addWeighted(final_frame_output, content_opacity, full_frame_blur_bg, 1 - content_opacity, 0)

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
    parser.add_argument("--tracking_responsiveness", type=float, default=0.2, help="For 'tracking' mode: how responsive the crop is to the current detected box (0.0-1.0). Lower values mean more smoothing. Default: 0.2")
    parser.add_argument(
        "--object_weights",
        type=str,
        default="face:1.0,person:0.8,default:0.5", # Default weights
        help="Comma-separated string of 'label:weight' pairs (e.g., \"face:1.0,person:0.8,default:0.5\"). "
             "Assigns importance weights to detected object classes. 'default' is used for unspecified classes."
    )
    parser.add_argument("--smoothing_window_size", type=int, default=5, help="For 'tracking' mode: number of previous frames to consider for smoothing camera motion (e.g., 3-10). Default: 5.")
    parser.add_argument("--tracking_deadzone_center_px", type=int, default=10, help="Tracking mode: min pixel change in detected interest center to trigger camera movement. Default: 10.")
    parser.add_argument("--tracking_deadzone_size_percent", type=float, default=0.05, help="Tracking mode: min percentage change (0.0-1.0) in detected interest size to trigger camera movement/zoom. Default: 0.05 (5%).")
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
            process_video(str(vid), str(out_path), args.ratio, args.mode,
                          args.enable_padding, args.blur_amount, args.content_opacity,
                          args.tracking_responsiveness, object_weights_map=parsed_weights,
                          smoothing_window_size=args.smoothing_window_size,
                          tracking_deadzone_center_px=args.tracking_deadzone_center_px,
                          tracking_deadzone_size_percent=args.tracking_deadzone_size_percent)
    else:
        parsed_weights = parse_object_weights(args.object_weights)
        process_video(args.input, args.output, args.ratio, args.mode,
                      args.enable_padding, args.blur_amount, args.content_opacity,
                      args.tracking_responsiveness, object_weights_map=parsed_weights,
                      smoothing_window_size=args.smoothing_window_size,
                      tracking_deadzone_center_px=args.tracking_deadzone_center_px,
                      tracking_deadzone_size_percent=args.tracking_deadzone_size_percent)


if __name__ == "__main__":
    main()
