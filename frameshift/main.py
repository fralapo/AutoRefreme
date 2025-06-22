"""Command-line interface for FrameShift."""
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque
import subprocess # Added for ffmpeg
import shutil # Added for shutil.which
import os # Added for os.remove and os.replace
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .utils.detection import Detector
from .utils.crop import union_boxes, compute_crop, calculate_weighted_interest_region
from .weights_parser import parse_object_weights

# Helper Functions for Padding
def map_blur_input_to_kernel(intensity: int) -> int:
    """Maps an intensity value (0-10) to an odd kernel size for GaussianBlur."""
    if not 0 <= intensity <= 10:
        intensity = np.clip(intensity, 0, 10)

    # Mappatura: 0->1, 1->5, 2->9, ..., 5->21, ..., 10->61
    kernel_map = [1, 5, 9, 13, 17, 21, 27, 33, 41, 51, 61]
    return kernel_map[intensity]

def parse_color_to_bgr(color_input: str) -> Tuple[int, int, int]:
    """
    Parses a color string (name or "(R,G,B)") to a BGR tuple.
    Defaults to black if parsing fails. BGR order for OpenCV.
    """
    color_input = color_input.strip().lower()
    predefined_colors_bgr = {
        "black": (0, 0, 0), "white": (255, 255, 255),
        "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
        "yellow": (0, 255, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255),
    }
    if color_input in predefined_colors_bgr:
        return predefined_colors_bgr[color_input]

    if color_input.startswith("(") and color_input.endswith(")"):
        try:
            rgb_str = color_input[1:-1].split(',')
            if len(rgb_str) == 3:
                r = np.clip(int(rgb_str[0].strip()), 0, 255)
                g = np.clip(int(rgb_str[1].strip()), 0, 255)
                b = np.clip(int(rgb_str[2].strip()), 0, 255)
                return (b, g, r) # Return as BGR
            else:
                print(f"Warning: Invalid RGB tuple format for color '{color_input}'. Expected 3 values (R,G,B).")
        except ValueError:
            print(f"Warning: Could not parse RGB numeric values for color '{color_input}'.")
        except Exception as e:
            print(f"Warning: Error parsing RGB color string '{color_input}': {e}")

    if color_input != "black": # Avoid double warning if default 'black' fails (should not happen with predefined)
        print(f"Warning: Color '{color_input}' not recognized or invalid format. Defaulting to black.")
    return (0, 0, 0) # Default to black

def get_cv2_interpolation_flag(interpolation_name: str) -> int:
    """Maps an interpolation name string to the corresponding OpenCV flag."""
    interpolation_name = interpolation_name.strip().lower()
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4  # Lanczos over 8x8 neighborhood
    }
    default_flag = cv2.INTER_LANCZOS4 # Default to Lanczos4 if name is unknown
    flag = mapping.get(interpolation_name, None)
    if flag is None:
        print(f"Warning: Unknown interpolation type '{interpolation_name}'. Defaulting to 'lanczos'.")
        return default_flag
    return flag

# FFmpeg Muxing Function
def mux_video_audio_with_ffmpeg(
    original_video_path: str,
    processed_video_path: str, # Video senza audio
    final_output_path: str,
    ffmpeg_exec_path: str # Percorso all'eseguibile di ffmpeg
) -> bool:
    """
    Combina il video processato (senza audio) con l'audio del video originale
    usando FFmpeg, salvando il risultato nel final_output_path.
    Restituisce True in caso di successo, False altrimenti.
    """
    cmd = [
        ffmpeg_exec_path,
        '-y',
        '-i', processed_video_path,
        '-i', original_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac', # Usiamo aac standard, -strict experimental potrebbe non essere necessario per versioni recenti
        # '-strict', 'experimental', # Rimosso per ora, può essere aggiunto se aac standard fallisce
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        final_output_path
    ]

    # print(f"DEBUG: Esecuzione comando FFmpeg: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')

        if result.returncode == 0:
            # print(f"DEBUG: FFmpeg muxing completato con successo per {final_output_path}")
            return True
        else:
            print(f"ERRORE: FFmpeg ha fallito per {final_output_path} (exit code {result.returncode}).")
            # Limitare la stampa di stdout/stderr se sono molto lunghi
            print(f"FFmpeg stdout (prime 500 chars):\n{result.stdout[:500]}")
            print(f"FFmpeg stderr (prime 500 chars):\n{result.stderr[:500]}")
            return False
    except FileNotFoundError:
        # Questo errore dovrebbe essere già stato catturato dal check iniziale in main()
        print(f"ERRORE: Eseguibile FFmpeg non trovato a '{ffmpeg_exec_path}'. Impossibile processare l'audio.")
        return False
    except Exception as e:
        print(f"ERRORE: Eccezione durante l'esecuzione di FFmpeg: {e}")
        if hasattr(e, 'stdout') and e.stdout: print(f"FFmpeg stdout (eccezione):\n{e.stdout.decode(errors='ignore')[:500]}")
        if hasattr(e, 'stderr') and e.stderr: print(f"FFmpeg stderr (eccezione):\n{e.stderr.decode(errors='ignore')[:500]}")
        return False


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
    apply_padding_flag: bool,
    padding_type_str: str,
    padding_color_str: str,
    blur_amount_param: int,
    output_target_height: int,
    interpolation_flag: int, # Nuovo parametro per il flag di interpolazione OpenCV
    content_opacity: float = 1.0,
    object_weights_map: Dict[str, float] = None,
) -> Optional[str]: # Restituisce il percorso del video temporaneo o None
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

        resized_content = cv2.resize(content_to_fit, (scaled_w, scaled_h), interpolation=interpolation_flag)

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
                content_final_size = cv2.resize(resized_content, (final_copy_w, final_copy_h), interpolation=interpolation_flag)
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

    # Calcola le dimensioni di output basate sull'altezza target e l'aspect ratio
    # aspect_ratio qui è W/H
    out_h = output_target_height
    out_w = int(round(out_h * aspect_ratio))

    # Assicura che out_w sia almeno 2 e pari (preferibile per molti codec)
    if out_w < 2:
        out_w = 2
    if out_w % 2 != 0:
        out_w += 1 # Rende out_w pari aggiungendo 1 se dispari (es. 719->720)
                   # Se out_w era 1 (improbabile), diventa 2.

    # print(f"DEBUG: Input dims: {width}x{height}, Target Output Dims: {out_w}x{out_h}, AR: {aspect_ratio}")

    # Crea un nome per il file video temporaneo (senza audio)
    temp_video_path = Path(output_path).parent / f"{Path(output_path).stem}_temp_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        print(f"ERROR: Could not open video writer for temporary file: {temp_video_path}")
        cap.release()
        return None

    # print(f"DEBUG: Writing temporary video to: {temp_video_path}")

    frame_idx = 0
    # prev_box è usato dalla modalità stationary per memorizzare il box fisso della scena.
    prev_box: Optional[np.ndarray] = None
    # pan_end non è più necessario

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
        # prev_box è stato calcolato all'inizio di process_video per la prima scena,
        # e viene ricalcolato sopra al cambio di scena.
        if prev_box is None:
            # Questo blocco è un fallback di sicurezza estrema, non dovrebbe essere raggiunto.
            # print("ERROR: prev_box is None in stationary mode unexpectedly at frame_idx:", frame_idx)
            current_crop_box_tuple = (0,0,width,height)
        else:
            current_crop_box_tuple = prev_box # prev_box è già un np.array(x,y,w,h) ma compute_crop resituisce tupla, quindi prev_box dovrebbe essere tupla
                                          # o dobbiamo assicurarci che sia sempre array e fare .astype(int) dopo.
                                          # sample_crop restituisce una tupla. Quindi prev_box (da np.array(sample_crop)) è un array.
                                          # Il fallback di sicurezza per crop_box is None sotto gestirà la conversione.

        # Assegnazione finale e unpacking di crop_box (che è current_crop_box_tuple o il fallback)
        # La variabile 'crop_box' è usata per coerenza con il blocco di fallback sotto.
        crop_box = current_crop_box_tuple

        if crop_box is None: # Fallback se, nonostante tutto, crop_box fosse None
            # print(f"WARNING: crop_box is None at frame {frame_idx} before unpacking. Defaulting.")
            # Questo caso dovrebbe essere ancora più raro ora.
            x1_crop, y1_crop, cw_crop, ch_crop = 0, 0, width, height
        elif isinstance(crop_box, np.ndarray):
            x1_crop, y1_crop, cw_crop, ch_crop = crop_box.astype(int)
        else: # Assumendo sia una tupla (x,y,w,h) da sample_crop o il fallback (0,0,w,h)
            x1_crop, y1_crop, cw_crop, ch_crop = map(int, crop_box)

        cropped_content = frame[y1_crop:y1_crop+ch_crop, x1_crop:x1_crop+cw_crop]

        cropped_content = frame[y1_crop:y1_crop+ch_crop, x1_crop:x1_crop+cw_crop]

        if not apply_padding_flag: # Corrisponde a padding_style='fill' se non specificato --padding
            # Logica "Fill" / "Pan & Scan"
            final_frame_output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            if cropped_content.shape[0] > 0 and cropped_content.shape[1] > 0:
                content_h, content_w = cropped_content.shape[:2]
                if content_h > 0 and content_w > 0:
                    scale_h_fill = out_h / content_h; scale_w_fill = out_w / content_w
                    scale_fill = max(scale_h_fill, scale_w_fill)
                    scaled_content_fill_w = int(content_w * scale_fill); scaled_content_fill_h = int(content_h * scale_fill)
                    if scaled_content_fill_w > 0 and scaled_content_fill_h > 0:
                        content_to_process = cv2.resize(cropped_content, (scaled_content_fill_w, scaled_content_fill_h), interpolation=interpolation_flag)
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
            # Se cropped_content è vuoto, final_frame_output rimane nero.

        else: # apply_padding_flag is True, usa padding_type_str
            base_frame_for_padding: Optional[np.ndarray] = None
            if padding_type_str == 'blur':
                kernel_size_for_padding_blur = map_blur_input_to_kernel(blur_intensity_0_10) # blur_intensity_0_10 è il nuovo nome di blur_amount_param
                base_frame_for_padding = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=interpolation_flag), (kernel_size_for_padding_blur, kernel_size_for_padding_blur), 0)
            elif padding_type_str == 'color':
                color_bgr = parse_color_to_bgr(padding_color_str)
                base_frame_for_padding = np.full((out_h, out_w, 3), color_bgr, dtype=np.uint8)
            else: # Default padding type è 'black' (o se tipo non riconosciuto)
                if padding_type_str != 'black':
                     print(f"Warning: Unknown padding_type '{padding_type_str}' with --padding. Defaulting to 'black'.")
                base_frame_for_padding = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            final_frame_output = _apply_fit_padding(base_frame_for_padding, cropped_content, out_w, out_h)

        # Applicazione Opacità (comune a tutti i casi)
        if content_opacity < 1.0:
            # Per lo sfondo dell'opacità, usiamo una sfocatura basata su blur_amount_param mappato a kernel
            # Questo per coerenza, dato che blur_amount ora è 0-10.
            opacity_bg_kernel_size = map_blur_input_to_kernel(blur_amount_param) # blur_amount_param è l'input 0-10 dall'utente
            full_frame_blur_bg_for_opacity = cv2.GaussianBlur(cv2.resize(frame, (out_w, out_h), interpolation=interpolation_flag),
                                                              (opacity_bg_kernel_size, opacity_bg_kernel_size), 0)
            final_frame_output = cv2.addWeighted(final_frame_output, content_opacity, full_frame_blur_bg_for_opacity, 1 - content_opacity, 0)

        out.write(final_frame_output)

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    # print(f"DEBUG: Finished writing temporary video: {temp_video_path}")
    return str(temp_video_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameShift auto reframing tool (stationary mode only).")

    # Check for ffmpeg dependency first
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print("WARNING: ffmpeg not found in PATH. Audio will not be processed or included in the output video.")
        print("         Please install FFmpeg and ensure it's in your system's PATH for audio support.")
    # else:
        # print(f"DEBUG: Found ffmpeg at: {ffmpeg_path}")

    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--ratio", default="9/16", help="Target aspect ratio (e.g., 9/16)")
    # --mode and tracking-specific arguments are removed.
    # --mode and tracking-specific arguments are removed.

    # Arguments for padding configuration
    parser.add_argument("--padding", action="store_true", default=False, help="Enable padding if content doesn't fill the frame. Default padding is black bars.")
    parser.add_argument(
        "--padding_type",
        type=str,
        default="black",
        choices=['black', 'blur', 'color'],
        help="Type of padding to use if --padding is enabled. 'black': Solid black bars (default if --padding is set). 'blur': Blurred background bars. 'color': Solid color bars specified by --padding_color_value."
    )
    parser.add_argument("--blur_amount", type=int, default=5, help="Blur intensity for padding (0-10, higher is more blur) if --padding_type='blur'. Default: 5.")
    parser.add_argument("--padding_color_value", type=str, default="black", help="Color for padding if --padding_type='color'. Accepts names (e.g., 'white', 'blue') or RGB tuples as string (e.g., \"(255,0,0)\" for red).")
    parser.add_argument("--output_height", type=int, default=1080, help="Target height for the output video (e.g., 720, 1080, 1280, 1920). Width is calculated based on the target aspect ratio. Default: 1080px.")
    parser.add_argument("--interpolation", type=str, default="lanczos", choices=['nearest', 'linear', 'cubic', 'area', 'lanczos'], help="Interpolation algorithm for resizing. 'lanczos' or 'cubic' for upscaling, 'area' for downscaling. Default: 'lanczos'.")

    parser.add_argument("--content_opacity", type=float, default=1.0, help="Opacity of the main content. If < 1.0, content is blended with a full-frame blurred background (applied after padding).")
    parser.add_argument(
        "--object_weights",
        type=str,
        default="face:1.0,person:0.8,default:0.5",
        help="Comma-separated 'label:weight' pairs (e.g., \"face:1.0,person:0.8,default:0.5\"). "
             "Assigns importance weights to detected objects. 'default' for unspecified classes."
    )
    parser.add_argument("--batch", action="store_true", help="Process all videos in input directory")

    args = parser.parse_args()

    if args.batch:
        in_dir = Path(args.input)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        parsed_weights = parse_object_weights(args.object_weights)
        interpolation_flag_cv2 = get_cv2_interpolation_flag(args.interpolation)

        for vid in in_dir.iterdir():
            if vid.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
                continue
            out_path = out_dir / f"{vid.stem}_reframed.mp4"
            # Il percorso originale del video di input è str(vid)
            # Il percorso finale desiderato è out_path
            temp_video_file = process_video(str(vid), str(out_path), args.ratio,
                                            apply_padding_flag=args.padding,
                                            padding_type_str=args.padding_type,
                                            padding_color_str=args.padding_color_value,
                                            blur_amount_param=args.blur_amount,
                                            output_target_height=args.output_height,
                                            interpolation_flag=interpolation_flag_cv2, # Aggiunto
                                            content_opacity=args.content_opacity,
                                            object_weights_map=parsed_weights)

            if temp_video_file:
                if ffmpeg_path:
                    # print(f"DEBUG: Attempting to mux audio for {out_path} using {temp_video_file}")
                    success = mux_video_audio_with_ffmpeg(str(vid), temp_video_file, str(out_path), ffmpeg_path)
                    if success:
                        try:
                            os.remove(temp_video_file)
                            # print(f"DEBUG: Removed temporary video file: {temp_video_file}")
                        except OSError as e:
                            print(f"Warning: Could not remove temporary video file {temp_video_file}: {e}")
                    else:
                        print(f"Warning: FFmpeg muxing failed for {str(out_path)}. Outputting video without audio.")
                        try:
                            # Rinomina il file temporaneo (senza audio) al nome di output finale
                            shutil.move(temp_video_file, str(out_path)) # shutil.move sovrascrive
                            print(f"Info: Video (no audio) saved to {str(out_path)}")
                        except OSError as e:
                            print(f"ERRORE: Could not rename temp video {temp_video_file} to {str(out_path)}: {e}")
                else: # ffmpeg non disponibile
                    # print(f"DEBUG: FFmpeg not available. Renaming {temp_video_file} to {str(out_path)}")
                    try:
                        shutil.move(temp_video_file, str(out_path))
                    except OSError as e:
                         print(f"ERRORE: Could not rename temp video {temp_video_file} to {str(out_path)}: {e}")
            else:
                print(f"ERROR: Video processing failed for {str(vid)}, no temporary file created.")

    else: # Single file mode
        parsed_weights = parse_object_weights(args.object_weights)
        # args.input è il video originale, args.output è il file finale desiderato
        interpolation_flag_cv2 = get_cv2_interpolation_flag(args.interpolation) # Anche per single file mode
        temp_video_file = process_video(args.input, args.output, args.ratio,
                                        apply_padding_flag=args.padding,
                                        padding_type_str=args.padding_type,
                                        padding_color_str=args.padding_color_value,
                                        blur_amount_param=args.blur_amount,
                                        output_target_height=args.output_height,
                                        interpolation_flag=interpolation_flag_cv2, # Aggiunto
                                        content_opacity=args.content_opacity,
                                        object_weights_map=parsed_weights)

        if temp_video_file:
            if ffmpeg_path:
                # print(f"DEBUG: Attempting to mux audio for {args.output} using {temp_video_file}")
                success = mux_video_audio_with_ffmpeg(args.input, temp_video_file, args.output, ffmpeg_path)
                if success:
                    try:
                        os.remove(temp_video_file)
                        # print(f"DEBUG: Removed temporary video file: {temp_video_file}")
                    except OSError as e:
                        print(f"Warning: Could not remove temporary video file {temp_video_file}: {e}")
                else:
                    print(f"Warning: FFmpeg muxing failed for {args.output}. Outputting video without audio.")
                    try:
                        shutil.move(temp_video_file, args.output)
                        print(f"Info: Video (no audio) saved to {args.output}")
                    except OSError as e:
                         print(f"ERRORE: Could not rename temp video {temp_video_file} to {args.output}: {e}")
            else: # ffmpeg non disponibile
                # print(f"DEBUG: FFmpeg not available. Renaming {temp_video_file} to {args.output}")
                try:
                    shutil.move(temp_video_file, args.output)
                except OSError as e:
                    print(f"ERRORE: Could not rename temp video {temp_video_file} to {args.output}: {e}")
        else:
            print(f"ERROR: Video processing failed for {args.input}, no temporary file created.")


if __name__ == "__main__":
    main()
