import DataFlow as df
import setup

IMG_HEIGHT = 240
IMG_WIDTH = 360
NUM_CHANNELS = 3

MODEL_PATH = ""
VIDEO_NAME = ""

paths = setup.setup()

root = paths[0]
metadata = paths[1]
vid_dir = paths[2]
frames_dir = paths[3]
intermediate_frames_dir = paths[4]
scale_down_frames_dir = paths[5]
input_train_frames_dir = paths[6]
output_train_frames_dir = paths[7]
input_training_dataset = paths[8]
output_training_dataset = paths[9]
mean_std_file = paths[10]

df.start_data_flow(vid_dir, frames_dir, scale_down_frames_dir, input_train_frames_dir, output_train_frames_dir,
                   input_training_dataset, output_training_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, mean_std_file, 0.25)
