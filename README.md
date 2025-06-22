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

```
pip install -r requirements.txt
```

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
2.  **Content Analysis:** Detects faces (MediaPipe) and objects (YOLOv8) within each scene, using `--object_weights` to determine their importance.
3.  **Optimal Stationary Crop:** For each scene, calculates a fixed crop window that best frames the weighted area of interest at the target aspect ratio.
4.  **Output Generation (Cropping/Padding):**
    *   **Default (Fill/Pan & Scan):** If `--padding` is NOT specified, the determined crop is scaled to completely fill the output frame. Excess parts of the crop are trimmed to match the target aspect ratio without deforming the image. No bars are added.
    *   **With Padding (`--padding` is specified):** The determined crop is scaled to fit entirely *within* the output frame, preserving its aspect ratio.
        *   If `--padding_type` is `black` (default when `--padding` is used) or an unrecognized type: Black bars are added.
        *   If `--padding_type` is `blur`: Blurred bars from the original video background are added. Intensity is controlled by `--blur_amount`.
        *   If `--padding_type` is `color`: Bars of the color specified by `--padding_color_value` are added.

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
*   `--content_opacity O`: (Default: `1.0`) Opacity of the main video content (0.0-1.0). If < 1.0, the content (including any padding) is blended with a blurred version of the full original frame.
*   `--object_weights "label:w,..."`: (Default: `"face:1.0,person:0.8,default:0.5"`) Comma-separated `label:weight` pairs for object importance (e.g., `face:1.0,person:0.8`).
*   `--batch`: (Flag) Process all videos in the input directory.

The cropping logic determines an optimal stationary (fixed) crop for each scene, prioritizing important content based on `--object_weights`. How this crop is presented in the final output (filled, or with padding) is controlled by `--padding` and its related arguments.

## License

FrameShift is released under the MIT License.
