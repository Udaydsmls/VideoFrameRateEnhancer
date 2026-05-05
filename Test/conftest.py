import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "clip.mp4"
    width, height, fps, frames = 64, 48, 10, 16
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV could not open mp4v writer in this environment")
    try:
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = (i * 3) % (width - 8)
            frame[10:18, x : x + 8] = (0, 255, 0)
            writer.write(frame)
    finally:
        writer.release()
    return video_path


@pytest.fixture
def frame_directory(tmp_path: Path) -> Path:
    folder = tmp_path / "frames" / "clip"
    folder.mkdir(parents=True)
    for i in range(8):
        frame = np.full((32, 32, 3), i * 30, dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_clip_{i:06d}.jpg"), frame)
    return folder.parent
