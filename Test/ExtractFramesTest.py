from pathlib import Path

import cv2
import pytest

from VideoOperations.ExtractingFrames import extract_directory, save_video_frames


def test_save_video_frames_writes_jpgs(synthetic_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "frames"
    count = save_video_frames(synthetic_video, output)
    written = sorted(output.glob("*.jpg"))
    assert count == len(written) > 0


def test_save_video_frames_scales(synthetic_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "frames_small"
    save_video_frames(synthetic_video, output, scale_factor=0.5)
    sample = cv2.imread(str(next(output.glob("*.jpg"))))
    assert sample.shape[:2] == (24, 32)


def test_save_video_frames_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        save_video_frames(tmp_path / "missing.mp4", tmp_path / "out")


def test_extract_directory_creates_subfolders(synthetic_video: Path, tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    target = videos_dir / "alpha.mp4"
    target.write_bytes(synthetic_video.read_bytes())

    counts = extract_directory(videos_dir, tmp_path / "frames")
    assert "alpha" in counts
    assert (tmp_path / "frames" / "alpha").is_dir()
    assert counts["alpha"] > 0
