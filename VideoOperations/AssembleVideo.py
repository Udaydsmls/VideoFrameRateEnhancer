import math
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_video_fps(video_path: Path) -> int:
    """Return the frame rate of ``video_path`` rounded up."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if fps <= 0 or math.isnan(fps):
        raise ValueError(f"Could not read fps from: {video_path}")
    return int(math.ceil(fps))


def assemble_video(frames_dir: Path, output_path: Path, fps: int) -> int:
    """Encode every image in ``frames_dir`` into a video at ``fps``."""
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    frame_files = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames in: {frames_dir}")

    first = cv2.imread(str(frame_files[0]))
    if first is None:
        raise FileNotFoundError(f"Could not read first frame: {frame_files[0]}")
    height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for: {output_path}")

    written = 0
    try:
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written
