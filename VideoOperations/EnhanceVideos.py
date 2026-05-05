from pathlib import Path

from VideoOperations.AssembleVideo import assemble_video, read_video_fps
from VideoOperations.ExtractingFrames import VIDEO_EXTENSIONS


def enhance_videos_frame_rate(
    interpolated_frames_dir: Path,
    enhanced_videos_dir: Path,
    videos_dir: Path,
) -> dict[str, Path]:
    """Encode each subfolder of interpolated frames as a video at 2x source fps."""
    interpolated_frames_dir = Path(interpolated_frames_dir)
    enhanced_videos_dir = Path(enhanced_videos_dir)
    videos_dir = Path(videos_dir)

    sources = {
        p.stem: p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }

    outputs: dict[str, Path] = {}
    for folder in sorted(p for p in interpolated_frames_dir.iterdir() if p.is_dir()):
        source = sources.get(folder.name)
        if source is None:
            continue
        fps = read_video_fps(source)
        output = enhanced_videos_dir / f"{folder.name}.mp4"
        assemble_video(folder, output, fps=2 * fps)
        outputs[folder.name] = output
    return outputs
