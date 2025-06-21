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

Additional options include `--blur` and `--overlay` to control background padding.

## License

FrameShift is released under the MIT License.
