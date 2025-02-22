import os
import shutil


def merge_folders(source_dir1: str, source_dir2: str, destination_dir: str) -> None:
    """
    Merges two source directories into a destination directory.
    If duplicate file names exist, they are renamed to avoid overwriting.

    :param source_dir1: First source directory.
    :param source_dir2: Second source directory.
    :param destination_dir: Directory where merged contents will be stored.
    """
    os.makedirs(destination_dir, exist_ok=True)

    for source_dir in (source_dir1, source_dir2):
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory '{source_dir}' does not exist or is not a directory.")
            continue

        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)

            if os.path.isfile(source_file):
                destination_file = get_unique_filename(destination_file)
                shutil.move(source_file, destination_file)


def get_unique_filename(file_path: str) -> str:
    """
    Generates a unique filename by appending a numerical suffix if the file already exists.

    :param file_path: Original file path.
    :return: Unique file path.
    """
    if not os.path.exists(file_path):
        return file_path

    base, ext = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1

    return f"{base}_{counter}{ext}"


def merge_subdirectories(source_dir1: str, source_dir2: str, destination_dir: str) -> None:
    """
    Merges subdirectories of two source directories into corresponding subdirectories in the destination.

    :param source_dir1: First source directory containing subdirectories.
    :param source_dir2: Second source directory containing subdirectories.
    :param destination_dir: Destination directory for merged subdirectories.
    """
    if not os.path.isdir(source_dir1):
        print(f"Warning: Source directory '{source_dir1}' does not exist or is not a directory.")
        return

    for subdirectory in os.listdir(source_dir1):
        merge_folders(
            os.path.join(source_dir1, subdirectory),
            os.path.join(source_dir2, subdirectory),
            os.path.join(destination_dir, subdirectory)
        )
        print(f"Successfully merged subdirectory '{subdirectory}' into '{subdirectory}'.")
