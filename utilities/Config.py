import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Resolved paths and hyperparameters for the pipeline."""

    root: Path
    videos: Path
    frames: Path
    interpolated_frames: Path
    enhanced_videos: Path
    checkpoints: Path

    architecture: str = "unet"
    scale_factor: float = 1.0
    image_size: tuple[int, int] | None = None
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    validation_split: float = 0.1
    num_workers: int = 2
    device: str = "auto"
    extra: dict = field(default_factory=dict)

    def ensure_directories(self) -> None:
        for p in (
            self.root,
            self.videos,
            self.frames,
            self.interpolated_frames,
            self.enhanced_videos,
            self.checkpoints,
        ):
            p.mkdir(parents=True, exist_ok=True)


_DEFAULTS: dict = {
    "absolute_path": "",
    "root_dir": "data",
    "videos_dir": "videos",
    "frames_dir": "frames",
    "interpolated_frames_dir": "interpolated_frames",
    "enhanced_videos_dir": "enhanced_videos",
    "checkpoints_dir": "checkpoints",
    "architecture": "unet",
    "scale_factor": 1.0,
    "image_size": None,
    "num_epochs": 10,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "validation_split": 0.1,
    "num_workers": 2,
    "device": "auto",
}


def load_config(path: str | os.PathLike[str] = "setup.json") -> Config:
    """Read ``setup.json`` and build a resolved :class:`Config`."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path

    with config_path.open("r", encoding="utf-8") as f:
        data = {**_DEFAULTS, **json.load(f)}

    base = Path(data["absolute_path"]) if data["absolute_path"] else config_path.parent
    root = base / data["root_dir"]

    image_size = data["image_size"]
    if image_size is not None:
        image_size = tuple(image_size)

    extra = {k: v for k, v in data.items() if k not in _DEFAULTS}

    return Config(
        root=root,
        videos=root / data["videos_dir"],
        frames=root / data["frames_dir"],
        interpolated_frames=root / data["interpolated_frames_dir"],
        enhanced_videos=root / data["enhanced_videos_dir"],
        checkpoints=root / data["checkpoints_dir"],
        architecture=data["architecture"],
        scale_factor=float(data["scale_factor"]),
        image_size=image_size,
        num_epochs=int(data["num_epochs"]),
        batch_size=int(data["batch_size"]),
        learning_rate=float(data["learning_rate"]),
        validation_split=float(data["validation_split"]),
        num_workers=int(data["num_workers"]),
        device=str(data["device"]),
        extra=extra,
    )
