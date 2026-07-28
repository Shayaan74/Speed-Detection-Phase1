"""Detect, track, and count vehicles in a local video file or a YouTube link.

Usage:
    uv run python vehicle_detector.py <video-file-or-youtube-url>

The annotated video is written next to this script in `output/` unless you pass
`--output`. Run with `--help` to see every option.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

# COCO class ids that YOLO uses for things that drive on roads. Anything not in
# this set (people, traffic lights, benches, ...) is thrown away after detection.
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 6, 7]  # bicycle, car, motorcycle, bus, train, truck

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def is_url(source: str) -> bool:
    """Return True when the source looks like a web link rather than a local file."""
    return source.startswith(("http://", "https://", "www."))


def download_video(url: str, download_dir: Path) -> Path:
    """Download a video from a URL with yt-dlp and return the saved file path."""
    import yt_dlp

    download_dir.mkdir(parents=True, exist_ok=True)
    options = {
        # Cap at 1080p: 4K footage is slow to process and rarely improves detection.
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "%(title).80s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


def pick_device(requested: str) -> str:
    """Resolve the compute device, preferring the Apple GPU when available."""
    if requested != "auto":
        return requested

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def summarize(class_history: dict[int, Counter]) -> Counter:
    """Collapse per-vehicle class votes into one final count per vehicle type.

    A tracked vehicle can be labelled inconsistently across frames (a van may flip
    between "car" and "truck"), so each vehicle is assigned the label it received
    most often over its lifetime. That keeps the totals to one entry per vehicle.
    """
    totals: Counter = Counter()
    for votes in class_history.values():
        winning_name, _ = votes.most_common(1)[0]
        totals[winning_name] += 1
    return totals


def process_video(
    source_path: Path,
    target_path: Path,
    model_name: str,
    confidence: float,
    iou: float,
    device: str,
) -> Counter:
    """Annotate every frame with tracked vehicle boxes and return the vehicle totals."""
    video_info = sv.VideoInfo.from_video_path(video_path=str(source_path))
    model = YOLO(model_name)

    tracker = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=confidence
    )

    # Scale line and text sizes to the video resolution so labels stay readable on
    # both a 480p clip and a 4K one.
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER,
    )

    class_history: dict[int, Counter] = defaultdict(Counter)
    frames = sv.get_video_frames_generator(source_path=str(source_path))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame in tqdm(frames, total=video_info.total_frames, desc="Processing"):
            result = model(
                frame,
                conf=confidence,
                iou=iou,
                classes=VEHICLE_CLASS_IDS,
                device=device,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections=detections)

            class_names = detections.data.get(
                "class_name", np.array([""] * len(detections))
            )
            for tracker_id, class_name in zip(
                detections.tracker_id, class_names, strict=True
            ):
                class_history[int(tracker_id)][str(class_name)] += 1

            labels = [
                f"#{tracker_id} {class_name}"
                for tracker_id, class_name in zip(
                    detections.tracker_id, class_names, strict=True
                )
            ]

            annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
            sink.write_frame(annotated)

    return summarize(class_history)


def main() -> None:
    """Parse command line arguments and run vehicle detection on the given video."""
    parser = argparse.ArgumentParser(
        description="Detect and count vehicles in a video file or YouTube link."
    )
    parser.add_argument("source", help="Path to a video file, or a video URL")
    parser.add_argument(
        "--output", help="Where to save the annotated video (default: output/<name>.mp4)"
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="YOLO model: yolo11n (fastest) < s < m < l < x (most accurate)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="How sure the model must be to report a vehicle, 0-1 (default: 0.3)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Overlap threshold used to merge duplicate boxes (default: 0.7)",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps, or a CUDA index like 0"
    )
    args = parser.parse_args()

    if is_url(args.source):
        print("Downloading video...")
        source_path = download_video(args.source, DEFAULT_OUTPUT_DIR / "downloads")
    else:
        source_path = Path(args.source)

    if not source_path.exists():
        parser.error(f"Video not found: {source_path}")

    if args.output:
        target_path = Path(args.output)
    else:
        target_path = DEFAULT_OUTPUT_DIR / f"{source_path.stem}-detected.mp4"

    device = pick_device(args.device)
    print(f"Reading  {source_path}")
    print(f"Model    {args.model} on {device}")

    totals = process_video(
        source_path=source_path,
        target_path=target_path,
        model_name=args.model,
        confidence=args.confidence,
        iou=args.iou,
        device=device,
    )

    print(f"\nSaved    {target_path}")
    print(f"\nUnique vehicles seen ({sum(totals.values())} total):")
    for class_name, count in totals.most_common():
        print(f"  {class_name:<12} {count}")


if __name__ == "__main__":
    main()
