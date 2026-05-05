from pathlib import Path

import cv2

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resize_image(input_path: Path, output_path: Path, scale_factor: float) -> None:
    """Resize a single image and save it as JPEG."""
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    h, w = image.shape[:2]
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale_factor))), max(1, int(round(h * scale_factor)))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized)


def batch_resize_images(input_folder: Path, output_folder: Path, scale_factor: float = 0.5) -> int:
    """Resize every image in ``input_folder`` into ``output_folder``."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in input_folder.iterdir():
        if entry.is_file() and entry.suffix.lower() in VALID_EXTENSIONS:
            resize_image(entry, output_folder / entry.name, scale_factor)
            written += 1
    return written


def resize_images_in_subfolders(input_folder: Path, output_folder: Path, scale_factor: float = 0.5) -> dict[str, int]:
    """Recursively resize images in each subdirectory of ``input_folder``."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    counts: dict[str, int] = {}
    for sub in sorted(p for p in input_folder.iterdir() if p.is_dir()):
        counts[sub.name] = batch_resize_images(sub, output_folder / sub.name, scale_factor)
    return counts
