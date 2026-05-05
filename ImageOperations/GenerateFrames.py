import shutil
from pathlib import Path

import cv2
import torch
from torch import nn

from ImageOperations.Dataset import IMAGE_EXTENSIONS, from_tensor, list_frames, load_image, to_tensor
from utilities.Checkpoints import find_latest_checkpoint, load_checkpoint
from utilities.Devices import resolve_device
from CreatingModel import build_model


@torch.no_grad()
def _interpolate_pair(model: nn.Module, prev: torch.Tensor, nxt: torch.Tensor, device: torch.device) -> torch.Tensor:
    return model(prev.unsqueeze(0).to(device), nxt.unsqueeze(0).to(device))[0].cpu()


def _interpolate_video_dir(model: nn.Module, source_dir: Path, output_dir: Path, device: torch.device) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = list_frames(source_dir)
    if len(frames) < 2:
        return 0

    prev_path = frames[0]
    prev_tensor = to_tensor(load_image(prev_path))
    shutil.copy2(prev_path, output_dir / prev_path.name)

    written = 0
    for current_path in frames[1:]:
        current_tensor = to_tensor(load_image(current_path))
        interpolated = _interpolate_pair(model, prev_tensor, current_tensor, device)
        cv2.imwrite(
            str(output_dir / f"{prev_path.stem}_interp.jpg"),
            cv2.cvtColor(from_tensor(interpolated), cv2.COLOR_RGB2BGR),
        )
        shutil.copy2(current_path, output_dir / current_path.name)
        prev_tensor = current_tensor
        prev_path = current_path
        written += 1
    return written


def generate_video_frames(
    frames_dir: Path,
    interpolated_frames_dir: Path,
    architecture: str,
    checkpoints_dir: Path,
    *,
    checkpoint: Path | None = None,
    device: str = "auto",
    model_kwargs: dict | None = None,
) -> dict[str, int]:
    """Run a trained interpolator over every video subfolder of ``frames_dir``."""
    frames_dir = Path(frames_dir)
    interpolated_frames_dir = Path(interpolated_frames_dir)
    torch_device = resolve_device(device)

    if checkpoint is None:
        checkpoint_path = find_latest_checkpoint(checkpoints_dir, architecture)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No '{architecture}' checkpoint under {checkpoints_dir}."
            )
    else:
        checkpoint_path = Path(checkpoint)

    ckpt = load_checkpoint(checkpoint_path, map_location=torch_device)
    if ckpt.architecture != architecture:
        raise ValueError(
            f"Checkpoint architecture '{ckpt.architecture}' does not match '{architecture}'."
        )
    model = build_model(architecture, **(model_kwargs or {}))
    model.load_state_dict(ckpt.state_dict)
    model.eval().to(torch_device)

    counts: dict[str, int] = {}
    for video_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        target = interpolated_frames_dir / video_dir.name
        counts[video_dir.name] = _interpolate_video_dir(model, video_dir, target, torch_device)
    return counts
