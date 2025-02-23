import json
import os


def get_paths(config_file: str = "setup.json") -> dict:
    """
    Return all the paths present in the configuration file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(script_dir, config_file), "r") as f:
        data = json.load(f)

    absolute_path = script_dir if len(data['absolute_path']) == 0 else data['absolute_path']
    root = os.path.join(absolute_path, data["root_dir"])

    paths = {
        "root": root,
        "metadata": os.path.join(root, data["metadata_dir"]),
        "vid_dir": os.path.join(root, data["vid_dir"]),
        "frames_dir": os.path.join(root, data["frames_dir"]),
        "intermediate_frames_dir": os.path.join(root, data["intermediate_frames_dir"]),
        "scale_down_frames_dir": os.path.join(root, data["scale_down_frames_dir"]),
        "input_train_frames_dir": os.path.join(root, data["input_train_frames_dir"]),
        "output_train_frames_dir": os.path.join(root, data["output_train_frames_dir"]),
        "input_training_dataset": os.path.join(root, data["input_training_dataset"]),
        "output_training_dataset": os.path.join(root, data["output_training_dataset"]),
        "enhanced_videos": os.path.join(root, data["enhanced_videos_dir"]),
        "models": os.path.join(root, data["trained_models"]),
        "dataset_dimensions": os.path.join(root, data["metadata_dir"], "dimensions"),
        "mean_std_file": os.path.join(root, data["metadata_dir"], f'{data["mean_std_file"]}.pkl'),
        ## is there even any point to have this; what to do about dimensions folder in metadata?
    }

    return paths


def get_values(config_file: str = "setup.json") -> dict:
    """
    Return all the values present in the configuration file.
    """
    with open(config_file, "r") as f:
        data = json.load(f)

    values = {
        "batch_size": data["batch_size_percent_int"],
        "scale_down_factor": data["scale_down_factor"],
    }

    return values


def setup(config_file: str = "setup.json") -> None:
    """
    Creates the necessary directories which are not present from the configuration file.
    """
    paths = get_paths(config_file)
    for path in paths:
        if path != "mean_std_file": os.makedirs(paths[path], exist_ok=True)
