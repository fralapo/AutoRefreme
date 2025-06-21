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

## Usage

```
python -m frameshift.main input.mp4 output.mp4 --ratio 9:16 --mode tracking
```

To process an entire directory of videos, use batch mode:

```
python -m frameshift.main my_videos out_dir --ratio 9:16 --mode stationary --batch
```

### Command-line Options

*   `input`: Path to the input video file or input directory (if `--batch` is used).
*   `output`: Path to the output video file or output directory (if `--batch` is used).
*   `--ratio R`: Target aspect ratio for the output video. Can be specified as `W:H` (e.g., `9:16`, `1:1`) or as a float (e.g., `0.5625`, `1.0`). Default: `9/16`.
*   `--mode M`: Cropping strategy. Default: `tracking`.
    *   `tracking`: Continuously tracks and centers detected faces/objects. The system aims to keep these elements within the frame, applying smoothing to the camera motion. This is the most dynamic mode.
    *   `stationary`: The cropping window remains fixed in an optimal position for each detected scene. Ideal for shots where the main content stays in a consistent area, similar to a tripod shot.
    *   `panning`: The cropping window moves at a constant speed from a determined start position to an end position within each scene. Good for slow, predictable movements or creating controlled panoramic effects.
*   `--enable_padding`: Enable letterbox (horizontal bars) or pillarbox (vertical bars) padding if the content's aspect ratio, after cropping for salient objects, doesn't match the target `--ratio`. The bars are filled with a blurred version of the video background.
*   `--blur_amount B`: Integer value (default: `21`) determining the kernel size for Gaussian blur. Used for:
    *   The background of padding bars if `--enable_padding` is active.
    *   The full-frame background if `--enable_padding` is not active but `--content_opacity` is less than `1.0`.
*   `--content_opacity O`: Float value between `0.0` (fully transparent) and `1.0` (fully opaque) (default: `1.0`). Controls the opacity of the main reframed video content.
    *   If `--enable_padding` is active, this makes the central content (inside the bars) transparent over the blurred bars.
    *   If `--enable_padding` is not active, this makes the reframed content transparent over a blurred version of the full original frame (resized to output dimensions).
*   `--batch`: Process all supported video files in the input directory. `input` and `output` are treated as directories.

The cropping logic inherently tries to maximize the visibility of the detected objects within the target aspect ratio, similar to a `MAXIMIZE_TARGET_DIMENSION` strategy.

## License

FrameShift is released under the MIT License.
