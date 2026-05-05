from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ImageOperations.ImageIO import (
    IMAGE_EXTENSIONS,
    from_tensor,
    list_frames,
    load_image,
    save_image,
    to_tensor,
)


def test_list_frames_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "ok.jpg").write_bytes(b"x")
    (tmp_path / "ok.png").write_bytes(b"x")
    (tmp_path / "skip.txt").write_bytes(b"x")
    (tmp_path / "skip.dat").write_bytes(b"x")
    files = list_frames(tmp_path)
    assert {p.suffix for p in files} <= IMAGE_EXTENSIONS
    assert len(files) == 2


def test_list_frames_missing_directory(tmp_path: Path) -> None:
    assert list_frames(tmp_path / "nope") == []


def test_load_image_resizes(tmp_path: Path) -> None:
    src = tmp_path / "img.jpg"
    cv2.imwrite(str(src), np.full((30, 40, 3), 200, dtype=np.uint8))
    image = load_image(src, image_size=(15, 20))
    assert image.shape == (15, 20, 3)
    assert 0.0 <= image.min() <= image.max() <= 1.0


def test_load_image_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")


def test_round_trip_to_tensor(tmp_path: Path) -> None:
    src = tmp_path / "img.jpg"
    cv2.imwrite(str(src), np.full((20, 20, 3), 128, dtype=np.uint8))
    image = load_image(src)
    tensor = to_tensor(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 20, 20)
    restored = from_tensor(tensor)
    assert restored.shape == image.shape
    assert restored.dtype == np.uint8


def test_save_image_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "out.jpg"
    save_image(target, np.full((8, 8, 3), 64, dtype=np.uint8))
    assert target.is_file()
