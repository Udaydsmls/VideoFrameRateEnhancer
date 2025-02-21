import unittest
import os
import cv2
import tempfile
import numpy as np
from VideoOperations import ExtractingFrames as ef


class TestExtractFrames(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.test_dir.name, "test_video.mp4")
        self.output_folder = os.path.join(self.test_dir.name, "output")

        # Create a dummy video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (640, 480)
        out = cv2.VideoWriter(self.video_path, fourcc, fps, frame_size)

        # Write a few frames
        for _ in range(5):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_extract_frames(self):
        ef.save_video_frames(self.video_path, self.output_folder)

        # Check if output directory is created
        video_name = "test_video"
        video_output_path = os.path.join(self.output_folder, video_name)
        self.assertTrue(os.path.exists(video_output_path))

        # Check if frames are extracted
        frame_files = [f for f in os.listdir(video_output_path) if f.endswith('.jpg')]
        self.assertGreater(len(frame_files), 0, "No frames were extracted")


if __name__ == "__main__":
    unittest.main()
