# FrameShift

FrameShift is an open source implementation inspired by Google AutoFlip. It automatically reframes
videos to a target aspect ratio while keeping faces and objects in view.

## Features

- Shot detection using PySceneDetect
- Face detection with MediaPipe
- Object detection using YOLOv8 (Ultralytics)
- Smooth camera path optimization to avoid jitter
- Output encoded with OpenCV

## Installation

1.  **Install Python Dependencies:**
    The script requires Python 3.8+ and the packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    This will install `opencv-python`, `mediapipe`, `scenedetect`, `ultralytics`, `huggingface_hub`, `tqdm`, and `numpy`.

2.  **Install FFmpeg (for audio processing):**
    FrameShift uses FFmpeg to process and include audio from the original video into the reframed output. If FFmpeg is not installed or not found in your system's PATH, the script will still process the video but the output will not contain audio (a warning will be displayed).
    *   **Download FFmpeg:** You can download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).
    *   **Installation:** Follow the installation instructions for your operating system. Ensure the directory containing the `ffmpeg` (or `ffmpeg.exe`) executable is added to your system's PATH environment variable.
    *   **Verify Installation:** Open a new terminal/command prompt and type `ffmpeg -version`. If it's installed correctly, you should see version information.

## Quick Start: Usage Examples

FrameShift automatically reframes videos, using a stationary (fixed) crop for each detected scene.

**1. Default Behavior: Fill Frame (Pan & Scan)**
   Converts a landscape video to portrait, filling the new 9:16 frame by cropping and centering.
   ```bash
   python -m frameshift.main my_landscape_video.mp4 video_for_stories.mp4 --ratio 9:16
   ```

**2. Padding with Black Bars:**
   If you want to see the entire optimally cropped region without parts being cut off to fill the frame, use `--padding`. This will add black bars by default.
   ```bash
   python -m frameshift.main input_video.mp4 square_video_fit.mp4 --ratio 1:1 --padding
   ```

**3. Padding with Blurred Background:**
   To use blurred background bars instead of black:
   ```bash
   python -m frameshift.main input_video.mp4 output_blur_padded.mp4 --ratio 16:9 --padding --padding_type blur --blur_amount 5
   ```
   *(Adjust `--blur_amount` from 0 (minimal) to 10 (maximal) for desired blur intensity.)*

**4. Padding with a Custom Color:**
   To use solid green bars:
   ```bash
   python -m frameshift.main input_video.mp4 output_color_padded.mp4 --ratio 4:3 --padding --padding_type color --padding_color_value green
   ```
   Or with a specific RGB color (e.g., blue):
   ```bash
   python -m frameshift.main input_video.mp4 output_rgb_padded.mp4 --ratio 4:3 --padding --padding_type color --padding_color_value "(0,0,255)"
   ```

**5. Batch Processing:**
   Reframes all videos in `my_video_folder` to 4:5, saving to `output_folder`, using default fill behavior.
   ```bash
   python -m frameshift.main my_video_folder/ output_folder/ --ratio 4:5 --batch
   ```

## How FrameShift Works (Inspired by Google AutoFlip)

FrameShift intelligently reframes videos using a **stationary (fixed) crop per scene** approach:

1.  **Scene Detection:** Divides the video into scenes using PySceneDetect.
2.  **Content Analysis:**
    *   **Faces:** Primarily detected using a specialized YOLOv8 model from Hugging Face (`arnabdhar/YOLOv8-Face-Detection`). MediaPipe is used as a fallback if the YOLO face model fails to load or during its initial download.
    *   **Other Objects:** Detected using the general-purpose YOLOv8n model. This step is performed *only if* you specify weights for object classes other than 'face' in the `--object_weights` argument. This optimizes performance when only face-centric reframing is needed.
    *   The importance of all detected elements in guiding the crop is determined by the `--object_weights` argument.
3.  **Optimal Stationary Crop:** For each scene, calculates a fixed crop window that best frames the weighted area of interest at the target aspect ratio.
4.  **Output Generation (Cropping/Padding):**
    *   **Default (Fill/Pan & Scan):** If `--padding` is NOT specified, the determined crop is scaled to completely fill the output frame. Excess parts of the crop are trimmed to match the target aspect ratio without deforming the image. No bars are added.
    *   **With Padding (`--padding` is specified):** The determined crop is scaled to fit entirely *within* the output frame, preserving its aspect ratio.
        *   If `--padding_type` is `black` (default when `--padding` is used) or an unrecognized type: Black bars are added.
        *   If `--padding_type` is `blur`: Blurred bars from the original video background are added. Intensity is controlled by `--blur_amount`.
        *   If `--padding_type` is `color`: Bars of the color specified by `--padding_color_value` are added.
    The script will attempt to preserve the audio track from the original video using FFmpeg. If FFmpeg is not found, the output video will be silent.

**Future Development Ideas:**
Enhancements could include automatic selection of reframing strategies (like dynamic tracking or panning) and more advanced path smoothing.

## Command-line Options

*   `input`: Path to the input video file or input directory (if `--batch` is used).
*   `output`: Path to the output video file or output directory (if `--batch` is used).
*   `--ratio R`: Target aspect ratio for the output video (e.g., `9:16`, `1:1`, or `0.5625`). Default: `9/16`.
*   `--padding`: (Flag, default: `False`) If set, enables padding to ensure the entire optimally cropped content is visible. Defaults to black bars if no other padding type is specified. If not set, the content will be cropped to fill the output frame (Pan & Scan).
*   `--padding_type TYPE`: (Default: `black`) Type of padding if `--padding` is enabled.
    *   `black`: Solid black bars.
    *   `blur`: Blurred background bars.
    *   `color`: Solid color bars (use with `--padding_color_value`).
*   `--blur_amount INT`: (Default: `5`) Integer from 0 (minimal) to 10 (maximal) for blur intensity when `padding_type` is `blur`.
*   `--padding_color_value STR`: (Default: `black`) Color for padding if `padding_type` is `color`.
    *   Accepts names: `black`, `white`, `red`, `green`, `blue`, `yellow`, `cyan`, `magenta`.
    *   Accepts RGB tuple as string: `"(R,G,B)"` (e.g., `"(255,0,0)"` for red).
*   `--output_height H`: (Default: `1080`, integer) Target height for the output video in pixels (e.g., 720, 1080, 1280, 1920). The width will be calculated automatically based on the target aspect ratio specified by `--ratio`. This allows control over the final output resolution.
*   `--interpolation METHOD`: (Default: `lanczos`) Specifies the interpolation algorithm used for resizing video frames.
    *   Choices: `nearest`, `linear`, `cubic`, `area`, `lanczos`.
    *   `lanczos` (Lanczos over 8x8 neighborhood) or `cubic` are generally recommended for upscaling (enlarging images) as they can produce sharper results.
    *   `area` is often best for downscaling (shrinking images).
    *   `linear` is faster but might be less sharp than `cubic` or `lanczos`.
    *   `nearest` is the fastest but produces blocky results, usually not recommended for video.
*   `--content_opacity O`: (Default: `1.0`) Opacity of the main video content (0.0-1.0). If < 1.0, the content (including any padding) is blended with a blurred version of the full original frame.
*   `--object_weights "label:w,..."`: (Default: `"face:1.0,person:0.8,default:0.5"`) Comma-separated `label:weight` pairs.
    *   Assigns importance weights to detected elements. The label `'face'` refers to faces detected by the specialized YOLOv8-Face model (or MediaPipe fallback). Other labels (e.g., `'person'`, `'car'`, `'dog'`) correspond to objects detected by YOLOv8n.
    *   YOLOv8n object detection is only run if weights are specified for labels other than `'face'` and `'default'` with a weight > 0.
    *   Example: `"--object_weights \"face:1.0,dog:0.7,default:0.2\""` (this would trigger YOLOv8n to look for dogs). If only `\"face:1.0,default:0.1\"` is given, YOLOv8n for general objects might not run if not explicitly needed.
*   `--batch`: (Flag) Process all videos in the input directory.

The cropping logic determines an optimal stationary (fixed) crop for each scene, prioritizing important content based on `--object_weights`. How this crop is presented in the final output (filled, or with padding) is controlled by `--padding` and its related arguments.

## License

FrameShift is released under the MIT License.
