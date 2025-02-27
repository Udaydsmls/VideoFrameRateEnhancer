import tensorflow as tf
from pathlib import Path

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def is_valid_image(filename: str) -> bool:
    """Check if a file has a valid image extension."""
    return filename.lower().endswith(VALID_EXTENSIONS)


def resize_image(input_path: Path, output_path: Path, scale_factor: float) -> None:
    """
    Resizes a single image and saves it to the output path.

    :param input_path: Path to the input image file.
    :param output_path: Path where the resized image will be saved.
    :param scale_factor: Factor by which the image will be resized.
    """
    image_data = tf.io.read_file(str(input_path))
    img = tf.image.decode_image(image_data, channels=3)

    new_size = tf.cast([tf.shape(img)[0] * scale_factor, tf.shape(img)[1] * scale_factor], tf.int32)

    img_resized = tf.image.resize(img, new_size, method=tf.image.ResizeMethod.LANCZOS3, antialias=True)
    img_resized = tf.cast(img_resized, tf.uint8)

    encoded_image = tf.io.encode_jpeg(img_resized)
    tf.io.write_file(str(output_path), encoded_image)


def batch_resize_images(input_folder: str, output_folder: str, scale_factor: float = 0.5) -> None:
    """
    Resizes all images in a folder by a given scale factor and stores them in the output folder.

    :param input_folder: Path to the folder containing images.
    :param output_folder: Path to the folder where resized images will be saved.
    :param scale_factor: Factor by which images will be resized.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.iterdir():
        if image_file.is_file() and is_valid_image(image_file.name):
            output_path = output_folder / image_file.name
            resize_image(image_file, output_path, scale_factor)


def resize_images_in_subfolders(input_folder: str, output_folder: str, scale_factor: float = 0.5) -> None:
    """
    Recursively resizes images in all subdirectories of a given folder.

    :param input_folder: Path to the folder containing multiple image directories.
    :param output_folder: Path where all the resized images will be stored in respective subdirectories.
    :param scale_factor: Factor by which images will be resized.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for subfolder in input_folder.iterdir():
        if subfolder.is_dir():
            batch_resize_images(str(subfolder), str(output_folder / subfolder.name), scale_factor)
            print(f"{subfolder.name} resized.")
