import json
import os


def setup():
    with open("setup.json", "r") as f:
        data = json.load(f)

    absolute_path = data["absolute_path"]
    root = data["root_dir"]

    if len(absolute_path) > 0:
        root = os.path.join(root, absolute_path)

    metadata = os.path.join(root, data["metadata_dir"])
    vid_dir = os.path.join(root, data["vid_dir"])
    frames_dir = os.path.join(root, data["frames_dir"])
    intermediate_frames_dir = os.path.join(root, data["intermediate_frames_dir"])
    scale_down_frames_dir = os.path.join(root, data["scale_down_frames_dir"])
    input_train_frames_dir = os.path.join(root, data["input_train_frames_dir"])
    output_train_frames_dir = os.path.join(root, data["output_train_frames_dir"])
    input_training_dataset = os.path.join(root, data["input_training_dataset"])
    output_training_dataset = os.path.join(root, data["output_training_dataset"])

    folders = [root, metadata, vid_dir, frames_dir, intermediate_frames_dir, scale_down_frames_dir,
               input_train_frames_dir,
               output_train_frames_dir, input_training_dataset, output_training_dataset]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    mean_std_file = os.path.join(metadata, f'{data["mean_std_file"]}.pkl')

    folders.append(mean_std_file)

    return folders
