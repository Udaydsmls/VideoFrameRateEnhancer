from pathlib import Path

import cv2
import numpy as np
import pytest

from VideoOperations.AssembleVideo import assemble_video, read_video_fps
from VideoOperations.EnhanceVideos import enhance_videos_frame_rate


def _write_frames(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        cv2.imwrite(
            str(folder / f"frame_{i:03d}.jpg"),
            np.full((24, 32, 3), i * 30, dtype=np.uint8),
        )


def test_read_video_fps(synthetic_video: Path) -> None:
    assert read_video_fps(synthetic_video) >= 1


def test_read_video_fps_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_video_fps(tmp_path / "no.mp4")


def test_assemble_video_writes_file(tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    _write_frames(folder, 6)
    output = tmp_path / "out.mp4"
    written = assemble_video(folder, output, fps=12)
    assert written == 6
    assert output.is_file() and output.stat().st_size > 0


def test_assemble_video_empty_folder(tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        assemble_video(folder, tmp_path / "out.mp4", fps=12)


def test_enhance_videos_matches_sources(synthetic_video: Path, tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clip.mp4").write_bytes(synthetic_video.read_bytes())

    interpolated_root = tmp_path / "interpolated"
    _write_frames(interpolated_root / "clip", 4)

    out = tmp_path / "enhanced"
    out.mkdir()
    outputs = enhance_videos_frame_rate(interpolated_root, out, videos)
    assert "clip" in outputs and outputs["clip"].is_file()


def test_enhance_videos_skips_unmatched_folder(synthetic_video: Path, tmp_path: Path, capsys) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "match.mp4").write_bytes(synthetic_video.read_bytes())

    interpolated_root = tmp_path / "interpolated"
    _write_frames(interpolated_root / "match", 3)
    _write_frames(interpolated_root / "stranger", 3)

    outputs = enhance_videos_frame_rate(interpolated_root, tmp_path / "enhanced", videos)
    assert set(outputs) == {"match"}
    captured = capsys.readouterr().out
    assert "stranger" in captured


def test_enhance_videos_fps_multiplier(synthetic_video: Path, tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clip.mp4").write_bytes(synthetic_video.read_bytes())

    interpolated_root = tmp_path / "interpolated"
    _write_frames(interpolated_root / "clip", 4)

    out = tmp_path / "enhanced"
    out.mkdir()
    outputs = enhance_videos_frame_rate(interpolated_root, out, videos, fps_multiplier=3)
    assert outputs["clip"].is_file()
