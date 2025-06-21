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
python -m frameshift.main input.mp4 output.mp4 --ratio 9:16
```

## License

FrameShift is released under the MIT License.
