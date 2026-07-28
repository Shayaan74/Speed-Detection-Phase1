# Speed-Detection-Phase1

Phase 1: detect, track, and count vehicles in a video file or a YouTube link.

Uses [YOLO11](https://docs.ultralytics.com) for detection and
[supervision](https://github.com/roboflow/supervision) for ByteTrack tracking and
annotation. Each tracked vehicle is counted once, using the class label it received
most often across its lifetime (a van that flips between "car" and "truck" still
counts as one vehicle).

## Setup

```bash
pip install -r requirements.txt
```

Model weights (`yolo11m.pt` by default) are downloaded automatically on first run.

## Usage

```bash
# local file
python vehicle_detector.py traffic.mp4

# youtube link
python vehicle_detector.py "https://www.youtube.com/watch?v=..."
```

The annotated video is written to `output/<name>-detected.mp4` unless you pass
`--output`.

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--output` | `output/<name>-detected.mp4` | Where to save the annotated video |
| `--model` | `yolo11m.pt` | YOLO model: `yolo11n` (fastest) < `s` < `m` < `l` < `x` (most accurate) |
| `--confidence` | `0.3` | How sure the model must be to report a vehicle (0-1) |
| `--iou` | `0.7` | Overlap threshold used to merge duplicate boxes |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or a CUDA index like `0` |

`auto` prefers Apple GPU (MPS), then CUDA, then CPU.

## Output

The console prints unique vehicle counts per class when processing finishes:

```
Unique vehicles seen (47 total):
  car          31
  truck         9
  bus           4
  motorcycle    3
```
