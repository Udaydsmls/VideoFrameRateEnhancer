from pathlib import Path

import pytest
import torch

from ImageOperations.Dataset import FrameTripletDataset, build_triplets
from ImageOperations.ImageIO import from_tensor, load_image, to_tensor


def test_build_triplets_counts(frame_directory: Path) -> None:
    triplets = build_triplets(frame_directory)
    assert len(triplets) == 6
    for t in triplets:
        assert t.prev != t.mid != t.next


def test_dataset_returns_tensors(frame_directory: Path) -> None:
    triplets = build_triplets(frame_directory)
    dataset = FrameTripletDataset(triplets)
    prev, nxt, mid = dataset[0]
    for tensor in (prev, nxt, mid):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape[0] == 3
        assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_dataset_image_size_resizes(frame_directory: Path) -> None:
    triplets = build_triplets(frame_directory)
    dataset = FrameTripletDataset(triplets, image_size=(16, 24))
    prev, _, _ = dataset[0]
    assert prev.shape == (3, 16, 24)


def test_empty_dataset_rejected() -> None:
    with pytest.raises(ValueError):
        FrameTripletDataset([])


def test_round_trip_tensor_image(frame_directory: Path) -> None:
    sample = next(iter((frame_directory / "clip").glob("*.jpg")))
    image = load_image(sample)
    restored = from_tensor(to_tensor(image))
    assert restored.shape == image.shape
