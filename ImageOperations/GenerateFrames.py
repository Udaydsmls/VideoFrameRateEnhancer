import os
import pickle
import numpy as np
import tensorflow as tf

from ConvertingData import load_and_preprocess_image
from ImageNormalization import denormalize_image


def load_model(model_path: str) -> tf.keras.Model:
    """Loads and returns the trained TensorFlow model."""
    return tf.keras.models.load_model(model_path)


def load_dataset_mean_std(mean_std_file: str = "MeanStdForDataset") -> tuple:
    """Loads the dataset mean and standard deviation from a pickle file."""
    with open(mean_std_file, "rb") as f:
        return pickle.load(f)


def process_frame_pair(model, first_frame, second_frame, mean, std) -> np.ndarray:
    """
    Generates a predicted frame using the model and denormalizes it.
    """
    first_frame = first_frame.reshape(1, *first_frame.shape)
    second_frame = second_frame.reshape(1, *second_frame.shape)

    prediction = model.predict([first_frame, second_frame])[0]
    prediction_denormalized = denormalize_image(prediction, mean, std)
    np.clip(prediction_denormalized, 0, 1, out=prediction_denormalized)

    return (prediction_denormalized * 255).astype(np.uint8)


def generate_video_frames(input_dir: str, model_path: str, output_dir: str, img_height: int, img_width: int,
                          num_channels: int) -> None:
    """
    Generates frames using a trained model and saves them to the output directory.

    :param input_dir: Directory containing input video frames.
    :param model_path: Path to the trained model.
    :param output_dir: Directory to save generated frames.
    :param img_height: Height of the images.
    :param img_width: Width of the images.
    :param num_channels: Number of image channels.
    """
    model = load_model(model_path)
    mean, std = load_dataset_mean_std()

    os.makedirs(output_dir, exist_ok=True)

    for video_folder in os.listdir(input_dir):
        video_input_path = os.path.join(input_dir, video_folder)
        video_output_path = os.path.join(output_dir, video_folder)

        os.makedirs(video_output_path, exist_ok=True)

        frame_files = sorted(
            [os.path.join(video_input_path, fname) for fname in os.listdir(video_input_path) if fname.endswith(".jpg")]
        )

        if not frame_files:
            continue  # Skip empty folders

        first_frame = load_and_preprocess_image(frame_files[0], img_height, img_width, num_channels, mean, std)
        first_frame_name = os.path.splitext(os.path.basename(frame_files[0]))[0]

        for i in range(len(frame_files) - 1):
            second_frame = load_and_preprocess_image(frame_files[i + 1], img_height, img_width, num_channels, mean, std)
            predicted_frame = process_frame_pair(model, first_frame, second_frame, mean, std)

            output_filename = os.path.join(video_output_path, f"{first_frame_name}_5.jpg")
            tf.io.write_file(output_filename, tf.image.encode_jpeg(predicted_frame))

            first_frame_name = os.path.splitext(os.path.basename(frame_files[i + 1]))[0]
            first_frame = second_frame  # Update for the next iteration
