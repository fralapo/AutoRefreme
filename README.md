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

FrameShift helps you automatically reframe videos to different aspect ratios. The reframing logic is based on a **stationary camera** approach per scene, meaning it finds an optimal fixed crop for each detected scene in your video.

**1. Convert a Landscape Video to Portrait (e.g., for Social Media Stories):**
   This example takes `my_landscape_video.mp4` and converts it to a 9:16 portrait format. The content will be cropped to fill the new aspect ratio, centering on the most important detected objects (like faces).

   ```bash
   python -m frameshift.main my_landscape_video.mp4 video_for_stories.mp4 --ratio 9:16
   ```
   *(This uses the default `--padding_style="fill"`)*

**2. Create a Square Video with Black Bar Padding:**
   This command reframes `input_video.mp4` to a 1:1 square. If the content (after being cropped to 1:1 optimally) doesn't fill the entire square (e.g., if the optimal 1:1 crop is smaller than the output resolution), black bars will be added.

   ```bash
   python -m frameshift.main input_video.mp4 square_video_black_bars.mp4 --ratio 1:1 --padding_style black
   ```

**3. Reframing with Blurred Padding (Letterbox/Pillarbox):**
   To make sure the optimally cropped content is fully visible and any necessary bars are blurred versions of the video background:

   ```bash
   python -m frameshift.main input_video.mp4 output_with_blur_padding.mp4 --ratio 16:9 --padding_style blur --blur_amount 35
   ```

**4. Batch Processing an Entire Directory:**
   To reframe all videos in `my_video_folder` to a 4:5 aspect ratio with the default "fill" behavior and save them to `output_folder`:

   ```bash
   python -m frameshift.main my_video_folder/ output_folder/ --ratio 4:5 --batch
   ```

## How FrameShift Works (Inspired by Google AutoFlip)

FrameShift aims to intelligently reframe videos by understanding their content. The current version focuses on a **stationary (fixed) crop per scene**, inspired by one of the modes in Google's AutoFlip.

1.  **Scene Detection (Shot Boundary Detection):**
    The video is first divided into individual scenes or "shots" using PySceneDetect's `ContentDetector`. Processing happens independently for each detected scene.

2.  **Content Analysis (Face and Object Detection):**
    Within each scene, FrameShift analyzes a sample of frames to identify important content:
    *   **Faces:** Detected using MediaPipe.
    *   **Common Objects:** Detected using a YOLOv8 model.
    The importance of these elements can be influenced using the `--object_weights` argument.

3.  **Optimal Stationary Crop per Scene:**
    For each scene, an optimal fixed cropping window is determined. This window aims to best frame the weighted area of interest (derived from faces and objects) throughout that scene, matching the target aspect ratio. This is like choosing the best fixed camera position for that entire shot.

4.  **Cropping, Resizing, and Padding (No-Deformation Policy):**
    FrameShift's primary goal is to reframe to the target aspect ratio **without deforming or stretching** the image. The `--padding_style` argument controls how this is achieved:
    *   **`fill` (Default):** The intelligently cropped content (which has the target aspect ratio) is scaled to completely fill the output frame dimensions. If the scaled content is larger than the output frame in one dimension (e.g., a wide 16:9 crop being fitted into a tall 9:16 output), it will be centered and the excess will be trimmed off (Pan & Scan). This ensures no bars and no image deformation, prioritizing a full frame of video.
    *   **`black`:** The intelligently cropped content is scaled to fit *within* the output frame dimensions while preserving its aspect ratio. If this results in empty areas (letterbox or pillarbox), these areas are filled with **black bars**.
    *   **`blur`:** Similar to `black`, but the empty areas are filled with a blurred version of the original video frame. The intensity of the blur is controlled by `--blur_amount`.

**Future Development Ideas (closer to full AutoFlip):**
While FrameShift currently focuses on a stationary crop per scene, future enhancements could include:
*   More advanced smoothing techniques for camera motion (if dynamic modes like tracking/panning are reintroduced).
*   Automatic selection of reframing strategies (`stationary`, `panning`, `tracking`) based on content analysis per scene.
*   More sophisticated criteria for "required" objects and how they influence padding decisions.

## Command-line Options

*   `input`: Path to the input video file or input directory (if `--batch` is used).
*   `output`: Path to the output video file or output directory (if `--batch` is used).
*   `--ratio R`: Target aspect ratio for the output video.
    *   Formats: `W:H` (e.g., `9:16`, `1:1`) or as a float representing width/height (e.g., `0.5625` for 9:16, `1.0` for 1:1).
    *   Default: `9/16`.
*   `--padding_style STYLE`: Defines how to handle the content if it doesn't perfectly fill the target aspect ratio after the optimal stationary crop is determined.
    *   `fill` (Default): Scales the cropped content to fill the output frame, potentially trimming edges (Pan & Scan). No bars are added.
    *   `black`: Scales the cropped content to fit entirely within the output frame, adding black bars if necessary (letterbox/pillarbox).
    *   `blur`: Scales the cropped content to fit entirely, adding blurred background bars if necessary.
    *   Default: `fill`.
*   `--blur_amount B`: Integer value (default: `21`). Kernel size for Gaussian blur when `padding_style` is `blur`. Also used for the full-frame background blur if `--content_opacity` is less than `1.0`.
*   `--content_opacity O`: Float value between `0.0` (fully transparent) and `1.0` (fully opaque) (default: `1.0`). Controls the opacity of the main reframed video content against its background (relevant if padding creates bars or if overall opacity is reduced).
*   `--object_weights "label1:weight1,label2:weight2,default:weight_default"`: (Default: `"face:1.0,person:0.8,default:0.5"`)
    *   Assigns importance weights to different object classes (e.g., 'face', 'person', 'car') for determining the center of interest for the stationary crop.
    *   Labels should be lowercase. Common labels from YOLO include 'person', 'car', 'dog', etc., plus 'face'.
    *   Weights are float values. Higher weights mean more importance.
    *   `default` weight applies to any detected object class not explicitly listed.
    *   Example: `"--object_weights \"face:1.0,person:0.9,dog:0.7,default:0.3\""`
*   `--batch`: If set, processes all supported video files in the `input` directory and saves them to the `output` directory.

The underlying cropping logic (for the stationary per-scene approach) always tries to maximize the visibility of the detected objects within the target aspect ratio, ensuring that the most important content (based on `--object_weights`) is prioritized for that fixed shot.

## License

FrameShift is released under the MIT License.
