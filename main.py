import os

import DataFlow as df

IMG_HEIGHT = 240
IMG_WIDTH = 360
NUM_CHANNELS = 3

ROOT_DIR = 'Data'
METADATA_DIR = os.path.join(ROOT_DIR, 'metadata')
VID_DIR = os.path.join(ROOT_DIR, 'Videos')
FRAMES_DIR = os.path.join(ROOT_DIR, 'Frames')
INTERMEDIATE_FRAMES_DIR = os.path.join(ROOT_DIR, 'IntermediateFrames')
SCALE_DOWN_FRAMES_DIR = os.path.join(ROOT_DIR, 'ScaleDown')
TRAIN_FRAMES_DIR = os.path.join(ROOT_DIR, 'TrainingFramesDataset')
TEST_FRAMES_DIR = os.path.join(ROOT_DIR, 'TestingFramesDataset')
TRAINING_DATASET_DIR = os.path.join(ROOT_DIR, 'TrainingDataset')
TESTING_DATASET_DIR = os.path.join(ROOT_DIR, 'TestingDataset')
MEAN_STD_file = os.path.join(METADATA_DIR, 'mean_std.pkl')
os.makedirs(METADATA_DIR, exist_ok=True)
MODEL_PATH = ""
VIDEO_NAME = ""



df.start_data_flow(ROOT_DIR, VID_DIR, FRAMES_DIR, SCALE_DOWN_FRAMES_DIR, TRAIN_FRAMES_DIR, TEST_FRAMES_DIR,
                TRAINING_DATASET_DIR, TESTING_DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, MEAN_STD_file)
