from pathlib import Path

import cv2
import numpy as np
import pytest

from VideoOperations.AssembleVideo import assemble_video, read_video_fps
from VideoOperations.EnhanceVideos import enhance_videos_frame_rate


def test_read_video_fps(synthetic_video: Path) -> None:
    assert read_video_fps(synthetic_video) >= 1


def test_read_video_fps_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_video_fps(tmp_path / "no.mp4")


def test_assemble_video_writes_file(tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(6):
        cv2.imwrite(
            str(folder / f"frame_{i:03d}.jpg"),
            np.full((24, 32, 3), i * 40, dtype=np.uint8),
        )
    output = tmp_path / "out.mp4"
    written = assemble_video(folder, output, fps=12)
    assert written == 6
    assert output.is_file() and output.stat().st_size > 0


def test_enhance_videos_matches_sources(synthetic_video: Path, tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    target = videos / "clip.mp4"
    target.write_bytes(synthetic_video.read_bytes())

    interpolated_root = tmp_path / "interpolated"
    folder = interpolated_root / "clip"
    folder.mkdir(parents=True)
    for i in range(4):
        cv2.imwrite(
            str(folder / f"frame_{i:03d}.jpg"),
            np.full((24, 32, 3), i * 60, dtype=np.uint8),
        )

    out = tmp_path / "enhanced"
    out.mkdir()
    outputs = enhance_videos_frame_rate(interpolated_root, out, videos)
    assert "clip" in outputs
    assert outputs["clip"].is_file()
