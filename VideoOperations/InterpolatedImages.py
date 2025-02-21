import os
import math
import cv2


def create_video_from_images(input_directory: str, output_video_path: str, frame_rate: int) -> None:
    """
    Creates a video from image frames stored in the specified directory.

    :param input_directory: Directory containing image frames (.jpg) to be combined.
    :param output_video_path: Path to save the generated video file.
    :param frame_rate: Frame rate for the output video.
    """
    frame_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".jpg")])

    if not frame_files:
        print("Error: No image frames found in the specified directory.")
        return

    first_frame = cv2.imread(os.path.join(input_directory, frame_files[0]))
    if first_frame is None:
        print("Error: Could not read the first frame.")
        return

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(input_directory, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Skipping unreadable frame {frame_file}")
            continue

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved successfully at: {output_video_path}")


def extract_video_frame_rate(video_path: str) -> int:
    """
    Extracts and returns the frame rate of a given video file.

    :param video_path: Path to the video file.
    :return: The frame rate of the video, rounded up to the nearest integer.
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return 0

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()

    return math.ceil(fps)