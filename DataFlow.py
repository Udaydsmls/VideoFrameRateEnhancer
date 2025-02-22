import shutil

import VideoOperations.ExtractingFrames as ef
import ImageOperations.ScaleDownImages as sd
import ImageOperations.ConvertingData as cd
import FolderOperations.MovingBackFiles as mf
import FolderOperations.TrainingTestingData as ttd
import os


def start_data_flow(vid_dir, frames_dir, scale_down_frames_dir, train_frames_dir, test_frames_dir,
                    training_dataset_dir, testing_dataset_dir, img_width, img_height, num_channels, mean_std_path, batch_size_percent):

    video_paths = [os.path.join(vid_dir, file_name) for file_name in os.listdir(vid_dir)]
    if len(video_paths) == 0:
        print("No video files found")
        return

    for video in video_paths:
        ef.save_video_frames(video, frames_dir)

    print('Frames extracted')

    sd.resize_images_in_subfolders(frames_dir, scale_down_frames_dir, scale_factor=0.25)

    print('Frames scaled')

    ttd.process_image_directories(scale_down_frames_dir, train_frames_dir, test_frames_dir, mean_std_path)

    print('Frames separated')

    cd.preprocess_video_frames(train_frames_dir, test_frames_dir, training_dataset_dir, testing_dataset_dir, mean_std_path, img_height,
                             img_width, num_channels, batch_size_percent)
    print('Frames processed')

    mf.merge_subdirectories(train_frames_dir, test_frames_dir, scale_down_frames_dir)

    print("Frames moved back to original folder")

    shutil.rmtree(train_frames_dir)
    shutil.rmtree(test_frames_dir)
