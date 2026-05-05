from pathlib import Path

import cv2
import numpy as np
import pytest

from ImageOperations.ScaleDownImages import (
    batch_resize_images,
    resize_image,
    resize_images_in_subfolders,
)


def _write_image(path: Path, height: int, width: int, value: int = 128) -> None:
    cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))


def test_resize_image_halves_dimensions(tmp_path: Path) -> None:
    src = tmp_path / "in.jpg"
    dst = tmp_path / "out.jpg"
    _write_image(src, 40, 60)
    resize_image(src, dst, scale_factor=0.5)
    out = cv2.imread(str(dst))
    assert out.shape[:2] == (20, 30)


def test_resize_image_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resize_image(tmp_path / "missing.jpg", tmp_path / "out.jpg", 0.5)


def test_batch_resize_images_writes_each(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        _write_image(src / f"f{i}.jpg", 20, 20, value=i * 10)
    out = tmp_path / "out"
    written = batch_resize_images(src, out, scale_factor=0.5)
    assert written == 3
    sample = cv2.imread(str(out / "f0.jpg"))
    assert sample.shape[:2] == (10, 10)


def test_resize_subfolders_processes_each_directory(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    _write_image(src / "a" / "f.jpg", 20, 20)
    _write_image(src / "b" / "f.jpg", 20, 20)
    out = tmp_path / "out"
    counts = resize_images_in_subfolders(src, out, scale_factor=0.5)
    assert counts == {"a": 1, "b": 1}
    assert (out / "a" / "f.jpg").is_file()
    assert (out / "b" / "f.jpg").is_file()
