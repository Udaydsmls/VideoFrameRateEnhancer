from FolderOperations import DataFlow as df
import setup

setup.setup()
paths = setup.get_paths()
values = setup.get_values()

root = paths["root"]
metadata = paths["metadata"]
vid_dir = paths["vid_dir"]
frames_dir = paths["frames_dir"]
intermediate_frames_dir = paths["intermediate_frames_dir"]
scale_down_frames_dir = paths["scale_down_frames_dir"]
input_train_frames_dir = paths["input_train_frames_dir"]
output_train_frames_dir = paths["output_train_frames_dir"]
input_training_dataset = paths["input_training_dataset"]
output_training_dataset = paths["output_training_dataset"]
mean_std_file = paths["mean_std_file"]

batch_size = values["batch_size"]
scale_down_factor = values["scale_down_factor"]

df.start_data_flow(vid_dir, frames_dir, scale_down_frames_dir, input_train_frames_dir, output_train_frames_dir,
                   input_training_dataset, output_training_dataset, batch_size, scale_down_factor)
