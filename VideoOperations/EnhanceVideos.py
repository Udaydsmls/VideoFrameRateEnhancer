from pathlib import Path

from VideoOperations.AssembleVideo import assemble_video, read_video_fps
from VideoOperations.ExtractingFrames import VIDEO_EXTENSIONS


def _index_source_videos(videos_dir: Path) -> dict[str, Path]:
    if not videos_dir.is_dir():
        return {}
    return {
        p.stem: p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }


def enhance_videos_frame_rate(
    interpolated_frames_dir: Path,
    enhanced_videos_dir: Path,
    videos_dir: Path,
    *,
    fps_multiplier: int = 2,
) -> dict[str, Path]:
    """Encode each subfolder of interpolated frames as a video at ``fps_multiplier`` x source fps."""
    interpolated_frames_dir = Path(interpolated_frames_dir)
    enhanced_videos_dir = Path(enhanced_videos_dir)
    videos_dir = Path(videos_dir)

    if not interpolated_frames_dir.is_dir():
        print(f"No interpolated frames at {interpolated_frames_dir}")
        return {}

    sources = _index_source_videos(videos_dir)
    outputs: dict[str, Path] = {}
    for folder in sorted(p for p in interpolated_frames_dir.iterdir() if p.is_dir()):
        source = sources.get(folder.name)
        if source is None:
            print(f"  {folder.name}: skipped (no matching source video in {videos_dir})")
            continue
        try:
            fps = read_video_fps(source)
        except Exception as exc:
            print(f"  {folder.name}: skipped ({exc})")
            continue
        output = enhanced_videos_dir / f"{folder.name}.mp4"
        assemble_video(folder, output, fps=fps_multiplier * fps)
        outputs[folder.name] = output
    return outputs
