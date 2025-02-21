import numpy as np
from PIL import Image


def compute_dataset_mean_std(image_paths: list[str]) -> (np.ndarray, np.ndarray):
    """
    Computes the mean and standard deviation for all the images in a dataset.

    :param image_paths: List of file paths to the images.
    :return: Tuple containing mean and standard deviation of the dataset.
    """
    sum_mean = np.zeros(3)
    sum_sq_mean = np.zeros(3)
    num_images = len(image_paths)

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0

        sum_mean += np.mean(img_array, axis=(0, 1))
        sum_sq_mean += np.mean(img_array ** 2, axis=(0, 1))

    mean = sum_mean / num_images
    std = np.sqrt(sum_sq_mean / num_images - mean ** 2)

    return mean, std


def normalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalizes an image using the given mean and standard deviation.

    :param image: A NumPy array representing the image.
    :param mean: The mean values of the dataset.
    :param std: The standard deviation values of the dataset.
    :return: Normalized image.
    """
    return (image - mean) / std


def denormalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalizes an image using the given mean and standard deviation.

    :param image: A NumPy array representing the normalized image.
    :param mean: The mean values of the dataset.
    :param std: The standard deviation values of the dataset.
    :return: Denormalized image with values clipped to [0, 1].
    """
    return np.clip(image * std + mean, 0, 1)
