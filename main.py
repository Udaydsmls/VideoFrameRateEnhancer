import DataFlow as df
import setup

IMG_HEIGHT = 240
IMG_WIDTH = 360
NUM_CHANNELS = 3

MODEL_PATH = ""
VIDEO_NAME = ""

setup.setup()
paths = setup.get_paths()

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

df.start_data_flow(vid_dir, frames_dir, scale_down_frames_dir, input_train_frames_dir, output_train_frames_dir,
                   input_training_dataset, output_training_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, mean_std_file, 0.25)
