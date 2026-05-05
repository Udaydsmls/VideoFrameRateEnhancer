from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Triplet:
    prev: Path
    mid: Path
    next: Path


def list_frames(folder: Path) -> list[Path]:
    """Sorted list of image files in ``folder``."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_triplets(frames_root: Path) -> list[Triplet]:
    """Walk ``frames_root`` and yield ``(prev, mid, next)`` triplets per video."""
    frames_root = Path(frames_root)
    triplets: list[Triplet] = []
    if not frames_root.is_dir():
        return triplets
    for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        frames = list_frames(video_dir)
        for i in range(len(frames) - 2):
            triplets.append(Triplet(frames[i], frames[i + 1], frames[i + 2]))
    return triplets


def load_image(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as a float32 RGB array in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image_size is not None:
        bgr = cv2.resize(bgr, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC float array -> CHW float tensor."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """CHW float tensor in [0, 1] -> HWC uint8 array."""
    arr = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255.0 + 0.5).astype(np.uint8)


class FrameTripletDataset(Dataset):
    """Dataset returning ``(prev, next, mid)`` tensors per item."""

    def __init__(self, triplets: Sequence[Triplet], image_size: tuple[int, int] | None = None) -> None:
        if not triplets:
            raise ValueError("FrameTripletDataset requires at least one triplet")
        self.triplets = list(triplets)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triplet = self.triplets[index]
        prev = to_tensor(load_image(triplet.prev, self.image_size))
        mid = to_tensor(load_image(triplet.mid, self.image_size))
        nxt = to_tensor(load_image(triplet.next, self.image_size))
        return prev, nxt, mid
