from pathlib import Path

from VideoOperations.ExtractingFrames import extract_directory


def start_data_flow(videos_dir: Path, frames_dir: Path, scale_factor: float = 1.0) -> dict[str, int]:
    """Extract frames from every video into per-video subfolders of ``frames_dir``."""
    videos_dir = Path(videos_dir)
    frames_dir = Path(frames_dir)
    if not videos_dir.is_dir() or not any(videos_dir.iterdir()):
        print(f"No videos found in '{videos_dir}'. Skipping extraction.")
        return {}
    counts = extract_directory(videos_dir, frames_dir, scale_factor=scale_factor)
    if not counts:
        print(f"No supported video files found in '{videos_dir}'.")
        return counts
    for name, n in counts.items():
        print(f"  {name}: {n} frames")
    return counts
