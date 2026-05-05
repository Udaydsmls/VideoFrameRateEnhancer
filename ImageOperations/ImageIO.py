from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_frames(folder: Path) -> list[Path]:
    """Return image files in ``folder`` sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load ``path`` as a float32 RGB array in [0, 1]."""
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


def save_image(path: Path, rgb_uint8: np.ndarray) -> None:
    """Persist an HWC uint8 RGB array as a JPEG via OpenCV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
