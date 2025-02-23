import shutil

import VideoOperations.ExtractingFrames as ef
import ImageOperations.ScaleDownImages as sd
import ImageOperations.ConvertingData as cd
import FolderOperations.MovingBackFiles as mf
import FolderOperations.SeparateData as ttd
import os


def start_data_flow(vid_dir: str, frames_dir: str, scale_down_frames_dir: str, input_frames_dir: str,
                    output_frames_dir: str, input_training_dataset_dir: str, output_training_dataset_dir: str,
                    batch_size_percent: int, scale_factor: float) -> bool:
    video_paths = [os.path.join(vid_dir, file_name) for file_name in os.listdir(vid_dir)]
    """
    Processes, formats and stores the data in it required location.
    """

    if len(video_paths) == 0:
        print("No video files found!!")
        print("Put the video files into your vid_dir folder from setup.json")
        print("Exiting...")
        return False

    for video in video_paths:
        ef.save_video_frames(video, frames_dir)

    sd.resize_images_in_subfolders(frames_dir, scale_down_frames_dir, scale_factor)

    ttd.process_image_directories(scale_down_frames_dir, input_frames_dir, output_frames_dir)

    cd.preprocess_video_frames(input_frames_dir, output_frames_dir, input_training_dataset_dir,
                               output_training_dataset_dir, batch_size_percent)

    mf.merge_subdirectories(input_frames_dir, output_frames_dir, scale_down_frames_dir)

    shutil.rmtree(input_frames_dir)
    shutil.rmtree(output_frames_dir)

    return True
