from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ImageOperations.ImageIO import list_frames, load_image, to_tensor


@dataclass(frozen=True)
class Triplet:
    prev: Path
    mid: Path
    next: Path


def build_triplets(frames_root: Path) -> list[Triplet]:
    """Walk per-video subfolders of ``frames_root`` and emit (prev, mid, next) triplets."""
    frames_root = Path(frames_root)
    if not frames_root.is_dir():
        return []
    triplets: list[Triplet] = []
    for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        frames = list_frames(video_dir)
        for i in range(len(frames) - 2):
            triplets.append(Triplet(frames[i], frames[i + 1], frames[i + 2]))
    return triplets


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
