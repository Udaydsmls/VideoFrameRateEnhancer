import cv2
import os


def save_video_frames(video_path: str, output_folder: str) -> None:
    """
    Extracts frames from a given video file and stores them in an output directory.

    Each video will have its own subdirectory named after the video file (without extension),
    where extracted frames will be saved as JPEG images.

    :param video_path: Path to the input video file.
    :param output_folder: Directory where extracted frames will be stored.
    """
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_folder, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_filename = f"frame_{video_name}_{frame_count:06d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    video_capture.release()
    print(f"Extraction completed: {frame_count} frames saved in '{video_output_dir}'.")
