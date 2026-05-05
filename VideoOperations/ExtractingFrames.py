from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def save_video_frames(video_path: Path, output_dir: Path, scale_factor: float = 1.0) -> int:
    """Extract every frame from ``video_path`` into ``output_dir`` as JPEG."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    name = video_path.stem
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if scale_factor != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (int(round(w * scale_factor)), int(round(h * scale_factor))),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            cv2.imwrite(str(output_dir / f"frame_{name}_{count:06d}.jpg"), frame)
            count += 1
    finally:
        capture.release()
    return count


def extract_directory(videos_dir: Path, frames_dir: Path, scale_factor: float = 1.0) -> dict[str, int]:
    """Extract frames for every video in ``videos_dir`` into per-video subfolders."""
    videos_dir = Path(videos_dir)
    frames_dir = Path(frames_dir)
    if not videos_dir.is_dir():
        return {}

    counts: dict[str, int] = {}
    for video in sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS):
        counts[video.stem] = save_video_frames(video, frames_dir / video.stem, scale_factor)
    return counts
