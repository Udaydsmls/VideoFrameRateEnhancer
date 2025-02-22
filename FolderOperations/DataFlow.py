import shutil

import VideoOperations.ExtractingFrames as ef
import ImageOperations.ScaleDownImages as sd
import ImageOperations.ConvertingData as cd
import FolderOperations.MovingBackFiles as mf
import FolderOperations.TrainingTestingData as ttd
import os


def start_data_flow(vid_dir: str, frames_dir: str, scale_down_frames_dir: str, train_frames_dir: str,
                    test_frames_dir: str, training_dataset_dir: str, testing_dataset_dir: str,
                    batch_size_percent: int, scale_factor: float) -> None:
    video_paths = [os.path.join(vid_dir, file_name) for file_name in os.listdir(vid_dir)]
    if len(video_paths) == 0:
        print("No video files found")
        return

    for video in video_paths:
        ef.save_video_frames(video, frames_dir)

    sd.resize_images_in_subfolders(frames_dir, scale_down_frames_dir, scale_factor)

    ttd.process_image_directories(scale_down_frames_dir, train_frames_dir, test_frames_dir)

    cd.preprocess_video_frames(train_frames_dir, test_frames_dir, training_dataset_dir, testing_dataset_dir,
                               batch_size_percent)

    mf.merge_subdirectories(train_frames_dir, test_frames_dir, scale_down_frames_dir)

    shutil.rmtree(train_frames_dir)
    shutil.rmtree(test_frames_dir)
