import unittest
import os
import shutil
from PIL import Image
from ImageOperations import ScaleDownImages as sd


class TestImageScaling(unittest.TestCase):
    def setUp(self):
        self.base_test_dir = "Test"
        os.makedirs(self.base_test_dir, exist_ok=True)

        self.input_folder = os.path.join(self.base_test_dir, "input")
        self.output_folder = os.path.join(self.base_test_dir, "output")

        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # Create a test image
        self.test_image_path = os.path.join(self.input_folder, "test.jpg")
        self.test_image = Image.new("RGB", (100, 100), color="red")

    def tearDown(self):
        shutil.rmtree(self.base_test_dir)

    def test_image_scale_down(self):
        self.test_image.save(self.test_image_path)

        sd.batch_resize_images(self.input_folder, self.output_folder, scale_factor=0.5)
        output_image_path = os.path.join(self.output_folder, 'test.jpg')

        self.assertTrue(os.path.exists(output_image_path))

        # Check if the image is correctly resized
        with Image.open(output_image_path) as img:
            self.assertEqual(img.size, (50, 50))

    def test_frames_scale_down(self):
        self.frame_folder = os.path.join(self.input_folder, "frames")
        os.makedirs(self.frame_folder, exist_ok=True)

        # Create a test image inside a frame folder
        self.test_frame_image_path = os.path.join(self.frame_folder, "frame1.jpg")
        self.test_image.save(self.test_frame_image_path)

        sd.resize_images_in_subfolders(self.input_folder, self.output_folder, scale_factor=0.5)

        output_frame_folder = os.path.join(self.output_folder, "frames")
        output_frame_image_path = os.path.join(output_frame_folder, "frame1.jpg")

        self.assertTrue(os.path.exists(output_frame_image_path))

        # Check if the image in frame folder is resized correctly
        with Image.open(output_frame_image_path) as img:
            self.assertEqual(img.size, (50, 50))


if __name__ == "__main__":
    unittest.main()
