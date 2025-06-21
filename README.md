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

FrameShift helps you automatically reframe videos to different aspect ratios. Here are some common use cases:

**1. Convert a Landscape Video to Portrait (e.g., for Social Media Stories):**
   This example takes a standard landscape video (`my_landscape_video.mp4`) and converts it to a 9:16 portrait format, ideal for platforms like Instagram Stories or TikTok. The `tracking` mode will attempt to keep the main subject in view.

   ```bash
   python -m frameshift.main my_landscape_video.mp4 video_for_stories.mp4 --ratio 9:16 --mode tracking
   ```

**2. Create a Square Video (e.g., for Instagram Feed):**
   This command reframes `input_video.mp4` to a 1:1 square aspect ratio.

   ```bash
   python -m frameshift.main input_video.mp4 square_video.mp4 --ratio 1:1 --mode tracking
   ```

**3. Reframing with Blurred Padding (Letterbox/Pillarbox):**
   If you want to ensure no part of your chosen crop is cut off and prefer blurred bars instead of black ones when the reframed content doesn't fill the target aspect ratio perfectly (even after intelligent cropping), use `--enable_padding`.

   ```bash
   python -m frameshift.main input_video.mp4 output_with_blur_padding.mp4 --ratio 16:9 --mode stationary --enable_padding --blur_amount 35
   ```
   *(Note: If the original video is already 16:9 and you target 16:9, padding might only appear if the detected content is very small and centered).*

**4. Batch Processing an Entire Directory:**
   To reframe all videos in a folder (`my_video_folder`) and save them to another folder (`output_folder`):

   ```bash
   python -m frameshift.main my_video_folder/ output_folder/ --ratio 4:5 --mode panning --batch
   ```

**5. Adjusting Tracking Smoothness:**
   If the `tracking` mode seems too jittery or too slow, you can adjust its responsiveness:

   ```bash
   python -m frameshift.main input.mp4 output.mp4 --ratio 9:16 --mode tracking --tracking_responsiveness 0.1
   ```
   *(A lower `--tracking_responsiveness` (e.g., 0.1) means more smoothing/slower camera. A higher value (e.g., 0.5) means faster response/less smoothing.)*


## How FrameShift Works (Inspired by Google AutoFlip)

FrameShift aims to intelligently reframe videos by understanding their content. The process is inspired by concepts from Google's AutoFlip and involves several key steps:

1.  **Scene Detection (Shot Boundary Detection):**
    The video is first divided into individual scenes or "shots." This is done by analyzing changes in color distribution between consecutive frames. Processing happens independently for each detected scene, allowing for different reframing strategies if needed. *(Currently uses PySceneDetect's ContentDetector).*

2.  **Content Analysis (Face and Object Detection):**
    Within each scene, FrameShift analyzes the frames to identify important content:
    *   **Faces:** Detected using MediaPipe.
    *   **Common Objects:** Detected using a YOLOv8 model (from Ultralytics).
    The areas containing these detected elements are considered important regions to keep in view. *(Currently, all detected objects/faces are given equal importance).*

3.  **Intelligent Reframing:**
    Based on the detected content and the chosen mode, a "virtual camera" decides how to frame each shot:
    *   **`stationary` mode:** The virtual camera finds an optimal fixed position for the entire scene, much like a camera on a tripod. This crop is determined by sampling frames within the scene and finding a common area of interest.
    *   **`panning` mode:** The virtual camera smoothly moves from a calculated start crop (based on content at the beginning of the scene) to an end crop (based on content at the end). This creates a panning effect. The amount of the scene sampled to determine these points can be adjusted (currently samples up to 25% of scene duration, min 30 frames, max 150 frames per endpoint).
    *   **`tracking` mode:** The virtual camera follows the detected faces/objects frame by frame. Smoothing is applied to prevent jittery movements, controlled by the `--tracking_responsiveness` parameter.

4.  **Cropping and Aspect Ratio Adherence (No-Deformation Policy):**
    FrameShift's primary goal is to reframe to the target aspect ratio **without deforming or stretching** the image.
    *   The core cropping logic (`compute_crop`) identifies the region of interest (e.g., detected faces/objects) and then calculates the largest possible rectangle that includes this region while matching the target aspect ratio.
    *   **Without Padding (`--enable_padding` is OFF):** If this intelligently cropped content (which already has the target aspect ratio) doesn't perfectly fill the final output dimensions (e.g., due to resolution differences), it's scaled while preserving its aspect ratio and centered. Any remaining space is filled with **black bars** (letterbox or pillarbox).
    *   **With Blurred Padding (`--enable_padding` is ON):** If you prefer, enabling this option will fill the letterbox/pillarbox areas with a blurred version of the original video frame instead of black bars. The intensity of the blur is controlled by `--blur_amount`.

**Future Development Ideas (closer to full AutoFlip):**
While FrameShift implements several core ideas, future enhancements could include:
*   More advanced smoothing techniques (e.g., polynomial path optimization).
*   Automatic selection of `stationary`, `panning`, or `tracking` modes based on content analysis.
*   Weighted importance for different types of detected objects.
*   More sophisticated criteria for "required" objects that must stay in frame.

## Command-line Options

*   `input`: Path to the input video file or input directory (if `--batch` is used).
*   `output`: Path to the output video file or output directory (if `--batch` is used).
*   `--ratio R`: Target aspect ratio for the output video.
    *   Formats: `W:H` (e.g., `9:16`, `1:1`) or as a float representing width/height (e.g., `0.5625` for 9:16, `1.0` for 1:1).
    *   Default: `9/16`.
*   `--mode M`: Cropping strategy. Default: `tracking`.
    *   `tracking`: Continuously tracks and centers detected faces/objects. Uses `--tracking_responsiveness` to control smoothness.
    *   `stationary`: Keeps the crop window fixed in an optimal position for each scene.
    *   `panning`: Pans the crop window smoothly from a start to an end position within each scene.
*   `--enable_padding`: If set, enables letterbox/pillarbox padding using a blurred version of the video background. If not set, black bars are used if padding is necessary to maintain aspect ratio without deformation.
*   `--blur_amount B`: Integer value (default: `21`). Kernel size for Gaussian blur. Affects:
    *   The blurred background for padding bars (if `--enable_padding` is active).
    *   The full-frame blurred background used when `--content_opacity` is less than `1.0` (this effect is applied on top of any existing padding).
*   `--content_opacity O`: Float value between `0.0` (fully transparent) and `1.0` (fully opaque) (default: `1.0`). Controls the opacity of the main reframed video content against its background (which could be blurred bars if `--enable_padding`, black bars, or a full blurred frame if opacity is < 1.0).
*   `--tracking_responsiveness TR`: Float value (0.0-1.0, default: `0.2`). For `tracking` mode only. Controls how quickly the camera reacts to detected object movements.
    *   Lower values (e.g., `0.1`) result in more smoothing and slower camera response (more inertia).
    *   Higher values (e.g., `0.5`, `0.8`) result in less smoothing and a faster, more direct camera response.
*   `--batch`: If set, processes all supported video files in the `input` directory and saves them to the `output` directory.

The underlying cropping logic always tries to maximize the visibility of the detected objects within the target aspect ratio (similar to a `MAXIMIZE_TARGET_DIMENSION` strategy), ensuring that the most important content is prioritized.

## License

FrameShift is released under the MIT License.
