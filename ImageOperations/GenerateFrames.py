import os
import glob
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

import ImageOperations.ConvertingData as cd
import ImageOperations.ImageNormalization as im
import FolderOperations.MovingBackFiles as mf
import utilities.utils as utils


def calculate_frames_mean_std(input_dir: str) -> tuple:
    """Calculates mean and standard deviation of the given frames"""
    all_image_paths = []

    for frame_folder in os.listdir(input_dir):
        source_path = os.path.join(input_dir, frame_folder)
        all_image_paths.extend(glob.glob(os.path.join(source_path, '*.jpg')))

    mean, std = im.compute_dataset_mean_std(all_image_paths)

    return mean, std


def process_frame_pair(model: tf.keras.models.Model, first_frame: np.ndarray, second_frame: np.ndarray,
                       mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Generates a predicted frame using the model and denormalizes it.
    """
    first_frame = first_frame.reshape(1, *first_frame.shape)
    second_frame = second_frame.reshape(1, *second_frame.shape)

    prediction = model.predict([first_frame, second_frame])[0]
    prediction_denormalized = im.denormalize_image(prediction, mean, std)
    np.clip(prediction_denormalized, 0, 1, out=prediction_denormalized)

    return (prediction_denormalized * 255).astype(np.uint8)


def generate_video_frames(input_dir: str, model_path: str, output_dir: str) -> None:
    """
    Generates frames using a trained model and saves them to the output directory.

    :param input_dir: Directory containing input video frames.
    :param model_path: Path to the trained model.
    :param output_dir: Directory to save generated frames.
    """

    model = tf.keras.models.load_model(model_path)
    mean, std = utils.load_mean_std_file()

    for video_folder in os.listdir(input_dir):
        video_input_path = os.path.join(input_dir, video_folder)
        video_output_path = os.path.join(output_dir, video_folder)

        os.makedirs(video_output_path, exist_ok=True)

        frame_files = sorted(
            [os.path.join(video_input_path, fname) for fname in os.listdir(video_input_path) if fname.endswith(".jpg")]
        )

        img_height, img_width, num_channels = cv2.imread(frame_files[0]).shape

        if not frame_files:
            continue

        j = max(len(os.listdir(video_output_path)) - 1, 0)

        first_frame = cd.load_and_preprocess_image(frame_files[j], img_height, img_width, num_channels, mean, std)
        first_frame_name = os.path.splitext(os.path.basename(frame_files[j]))[0]

        for i in range(j, len(frame_files) - 1):
            second_frame = cd.load_and_preprocess_image(frame_files[i + 1], img_height, img_width, num_channels, mean,
                                                        std)
            predicted_frame = process_frame_pair(model, first_frame, second_frame, mean, std)

            output_filename = os.path.join(video_output_path, f"{first_frame_name}_5.jpg")
            tf.io.write_file(output_filename, tf.image.encode_jpeg(predicted_frame))

            first_frame_name = os.path.splitext(os.path.basename(frame_files[i + 1]))[0]
            first_frame = second_frame

            K.clear_session()
            gc.collect()

    mf.merge_subdirectories(input_dir, output_dir, input_dir)

    shutil.rmtree(output_dir)
