import re
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Checkpoint:
    architecture: str
    state_dict: dict
    epoch: int
    metadata: dict


_CHECKPOINT_PATTERN = re.compile(r"^(?P<arch>[a-zA-Z0-9_]+)_epoch(?P<epoch>\d+)\.pt$")


def checkpoint_path(directory: Path, architecture: str, epoch: int) -> Path:
    """Build the canonical path for a checkpoint."""
    return Path(directory) / f"{architecture}_epoch{epoch:04d}.pt"


def save_checkpoint(checkpoint: Checkpoint, path: Path) -> Path:
    """Persist a checkpoint under ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": checkpoint.architecture,
            "state_dict": checkpoint.state_dict,
            "epoch": checkpoint.epoch,
            "metadata": checkpoint.metadata,
        },
        path,
    )
    return path


def load_checkpoint(path: Path, map_location: torch.device | str | None = None) -> Checkpoint:
    """Load a previously saved checkpoint."""
    raw = torch.load(Path(path), map_location=map_location)
    return Checkpoint(
        architecture=raw["architecture"],
        state_dict=raw["state_dict"],
        epoch=raw.get("epoch", 0),
        metadata=raw.get("metadata", {}),
    )


def find_latest_checkpoint(directory: Path, architecture: str | None = None) -> Path | None:
    """Return the highest-epoch checkpoint, optionally filtered by architecture."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = _CHECKPOINT_PATTERN.match(entry.name)
        if not match:
            continue
        if architecture is not None and match.group("arch") != architecture:
            continue
        epoch = int(match.group("epoch"))
        if best is None or epoch > best[0]:
            best = (epoch, entry)
    return best[1] if best else None
