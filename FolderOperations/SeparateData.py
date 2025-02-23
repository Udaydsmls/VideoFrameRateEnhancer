import os
import shutil
import re
import pickle
import glob
from typing import List
import ImageOperations.ImageNormalization as im
import setup


def ensure_directory_exists(path: str) -> None:
    """
    Creates a directory if it does not already exist.
    :param path: Path of the directory to be created.
    """
    os.makedirs(path, exist_ok=True)


def relocate_image(filename: str, source_folder: str, destination_folder: str) -> None:
    """
    Moves an image from the source folder to the destination folder.
    :param filename: Name of the file to be moved.
    :param source_folder: Directory where the image is currently stored.
    :param destination_folder: Directory where the image should be moved.
    """
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    shutil.move(source_path, destination_path)


def distribute_images(source_folder: str, input_train_folder: str, output_train_folder: str) -> None:
    """
    Separates images into training and testing sets by moving alternate images.
    Odd-numbered frames are moved to the training folder, even-numbered frames to the test folder.

    :param source_folder: Source directory containing all images of a single video.
    :param input_train_folder: Directory for storing images with odd frame numbers.
    :param output_train_folder: Directory for storing images with even frame numbers.
    """
    ensure_directory_exists(input_train_folder)
    ensure_directory_exists(output_train_folder)

    pattern = re.compile(r"frame_(.+)_(\d{6})\.(\w+)")

    for filename in os.listdir(source_folder):
        match = pattern.match(filename)
        if match:
            frame_number = int(match.group(2))
            target_folder = input_train_folder if frame_number % 2 == 0 else output_train_folder
            relocate_image(filename, source_folder, target_folder)


def gather_image_paths(directory: str) -> List[str]:
    """
    Collects all image file paths from a given directory.
    :param directory: Path of the directory to search for image files.
    :return: List of image file paths.
    """
    return glob.glob(os.path.join(directory, '*.jpg'))


def process_image_directories(source_folder: str, input_train_folder: str, output_train_folder: str) -> None:
    """
    Separates all image directories into training and testing sets.
    Processes each directory in the source folder and moves images accordingly.

    :param source_folder: Directory containing multiple image directories.
    :param input_train_folder: Directory for storing training image directories.
    :param output_train_folder: Directory for storing test image directories.
    :param output_pickle: Path to store the computed mean and standard deviation of the dataset.
    """
    all_image_paths = []

    for frame_folder in os.listdir(source_folder):
        source_path = os.path.join(source_folder, frame_folder)
        train_path = os.path.join(input_train_folder, frame_folder)
        test_path = os.path.join(output_train_folder, frame_folder)

        distribute_images(source_path, train_path, test_path)

        all_image_paths.extend(gather_image_paths(train_path))
        all_image_paths.extend(gather_image_paths(test_path))

        print(f"Files separated in {frame_folder}")

    mean, std = im.compute_dataset_mean_std(all_image_paths)

    paths = setup.get_paths()

    with open(paths['mean_std_file'], 'wb') as f:
        pickle.dump((mean, std), f)
