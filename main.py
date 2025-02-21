import os

from DataFlow import start_data_flow
from EnhanceVideo import create_video

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
MODEL_PATH = ""
VIDEO_NAME = ""



start_data_flow(ROOT_DIR, VID_DIR, FRAMES_DIR, SCALE_DOWN_FRAMES_DIR, TRAIN_FRAMES_DIR, TEST_FRAMES_DIR,
                TRAINING_DATASET_DIR, TESTING_DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

create_video(SCALE_DOWN_FRAMES_DIR, MODEL_PATH, INTERMEDIATE_FRAMES_DIR, VIDEO_NAME, VID_DIR, IMG_HEIGHT, IMG_WIDTH,
             NUM_CHANNELS)
