import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from ImageOperations.ImageNormalization import normalize_image


def load_image(image_path: str, img_height: int, img_width: int, num_channels: int) -> np.ndarray:
    """
    Load and decode an image from a file.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize(image, [img_height, img_width])
    return image / 255.0


def preprocess_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.array:
    """
    Normalize an image using mean and standard deviation.
    """
    return normalize_image(image, mean, std)


def load_and_preprocess_image(image_path: str, img_height: int, img_width: int, num_channels: int, mean: np.ndarray,
                              std: np.ndarray) -> np.ndarray:
    """
    Load, normalize, and preprocess an image.
    """
    image = load_image(image_path, img_height, img_width, num_channels)
    return preprocess_image(image, mean, std)


def save_preprocessed_data(folder: str, filename: str, data_key: str, data: np.ndarray) -> None:
    """
    Save preprocessed image data in compressed .npz format.
    """
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, filename), **{data_key: data})


def preprocess_dataset(input_folder: str, output_folder: str, processed_input_folder: str, processed_output_folder: str,
                       mean_std_path: str, img_height: int, img_width: int, num_channels: int,
                       batch_size_percent: int = 1) -> None:
    """
    Preprocess images from input and output folders and store them in batches.
    """
    base_folder = os.path.basename(input_folder)

    input_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg")])
    output_paths = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".jpg")])

    paired_input_paths = [[input_paths[i], input_paths[i + 1]] for i in range(len(input_paths) - 1)]
    min_length = min(len(paired_input_paths), len(output_paths))
    paired_input_paths, output_paths = shuffle(paired_input_paths[:min_length], output_paths[:min_length],
                                               random_state=42)

    with open(mean_std_path, "rb") as f:
        mean, std = pickle.load(f)

    batch_size = max(1, (len(paired_input_paths) * batch_size_percent) // 100)

    for i in range(0, len(paired_input_paths), batch_size):
        input_chunk = paired_input_paths[i:i + batch_size]
        output_chunk = output_paths[i:i + batch_size]

        input_data = np.array([
            [load_and_preprocess_image(img1, img_height, img_width, num_channels, mean, std),
             load_and_preprocess_image(img2, img_height, img_width, num_channels, mean, std)]
            for img1, img2 in input_chunk
        ])

        output_data = np.array([
            load_and_preprocess_image(img, img_height, img_width, num_channels, mean, std) for img in output_chunk
        ])

        save_preprocessed_data(processed_input_folder, f"trainData_{base_folder}_{i:06d}.npz", "train", input_data)
        save_preprocessed_data(processed_output_folder, f"testData_{base_folder}_{i:06d}.npz", "test", output_data)


def preprocess_video_frames(input_train_folder: str, output_train_folder: str, processed_input_folder: str,
                            processed_output_folder: str,
                            img_height: int, img_width: int, num_channels: int, batch_size_percent: int = 1) -> None:
    """
    Preprocess frames from multiple video directories.
    """
    for folder in os.listdir(input_train_folder):
        input_path = os.path.join(input_train_folder, folder)
        output_path = os.path.join(output_train_folder, folder)
        processed_input_path = os.path.join(processed_input_folder, folder)
        processed_output_path = os.path.join(processed_output_folder, folder)

        preprocess_dataset(input_path, output_path, processed_input_path, processed_output_path, img_height, img_width,
                           num_channels, batch_size_percent)
