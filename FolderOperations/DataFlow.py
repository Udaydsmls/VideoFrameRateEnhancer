from dataclasses import dataclass, field
from pathlib import Path

from VideoOperations.ExtractingFrames import VIDEO_EXTENSIONS, save_video_frames


@dataclass
class DataFlowResult:
    """Outcome of a data-flow run."""

    extracted: dict[str, int] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.extracted) and not self.failed


def _videos_in(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def start_data_flow(
    videos_dir: Path,
    frames_dir: Path,
    *,
    scale_factor: float = 1.0,
    skip_existing: bool = True,
) -> DataFlowResult:
    """Extract every video in ``videos_dir`` into per-video subfolders of ``frames_dir``.

    Each video is processed independently; failures are reported but do not
    abort the whole run. When ``skip_existing`` is true a video whose
    output folder already contains frames is left untouched, making the
    step idempotent.
    """
    videos_dir = Path(videos_dir)
    frames_dir = Path(frames_dir)
    result = DataFlowResult()

    if not videos_dir.is_dir():
        print(f"Videos directory does not exist: {videos_dir}")
        return result

    videos = _videos_in(videos_dir)
    if not videos:
        print(f"No supported video files found in '{videos_dir}'.")
        return result

    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(videos)} video(s) in {videos_dir}.")
    for video in videos:
        target = frames_dir / video.stem
        if skip_existing and _has_frames(target):
            print(f"  {video.name}: skipped (frames already exist)")
            result.extracted[video.stem] = _count_frames(target)
            continue
        try:
            count = save_video_frames(video, target, scale_factor=scale_factor)
        except Exception as exc:
            result.failed[video.stem] = str(exc)
            print(f"  {video.name}: FAILED ({exc})")
            continue
        result.extracted[video.stem] = count
        print(f"  {video.name}: {count} frames")
    return result


def _has_frames(folder: Path) -> bool:
    return folder.is_dir() and any(folder.iterdir())


def _count_frames(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file())
