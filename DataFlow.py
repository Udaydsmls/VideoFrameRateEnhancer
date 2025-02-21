import shutil

from ConvertingData_old import preprocessing_frame_data
from VideoOperations import ExtractingFrames as ef
from ImageOperations import ScaleDownImages as sd
from FolderOperations import MovingBackFiles as mf, TrainingTestingData as ttd
import os


def start_data_flow(root_dir, vid_dir, frames_dir, scale_down_frames_dir, train_frames_dir, test_frames_dir,
                    training_dataset_dir, testing_dataset_dir, img_width, img_height, num_channels):
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)
        return

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(scale_down_frames_dir, exist_ok=True)
    os.makedirs(train_frames_dir, exist_ok=True)
    os.makedirs(test_frames_dir, exist_ok=True)
    os.makedirs(training_dataset_dir, exist_ok=True)
    os.makedirs(testing_dataset_dir, exist_ok=True)

    video_paths = [os.path.join(vid_dir, file_name) for file_name in os.listdir(vid_dir)]
    for video in video_paths:
        ef.save_video_frames(video, frames_dir)

    print('Frames extracted')

    sd.resize_images_in_subfolders(frames_dir, scale_down_frames_dir, scale_factor=0.25)

    print('Frames scaled')

    ttd.process_image_directories(scale_down_frames_dir, train_frames_dir, test_frames_dir)

    print('Frames separated')

    preprocessing_frame_data(train_frames_dir, test_frames_dir, training_dataset_dir, testing_dataset_dir, img_height,
                             img_width,
                             num_channels)
    print('Frames processed')

    mf.merge_subdirectories(train_frames_dir, test_frames_dir, scale_down_frames_dir)

    print("Frames moved back to original folder")

    shutil.rmtree(train_frames_dir)
    shutil.rmtree(test_frames_dir)


if __name__ == '__main__':
    start_data_flow()
