import unittest
import numpy as np
import tensorflow as tf
import pickle
from unittest.mock import patch, mock_open
from ImageOperations import ConvertingData as cd


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.image_path = "test_image.jpg"
        self.img_height = 64
        self.img_width = 64
        self.num_channels = 3
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.2, 0.2, 0.2])
        self.folder = "test_folder"
        self.filename = "test_file.npz"
        self.data_key = "test_data"
        self.data = np.random.rand(10, 64, 64, 3)

    @patch("tensorflow.io.read_file")
    @patch("tensorflow.image.decode_jpeg")
    @patch("tensorflow.image.resize")
    def test_load_image(self, mock_resize, mock_decode, mock_read):
        mock_read.return_value = b"image_data"
        mock_decode.return_value = tf.random.uniform((100, 100, 3))
        mock_resize.return_value = tf.random.uniform((self.img_height, self.img_width, self.num_channels))

        image = cd.load_image(self.image_path, self.img_height, self.img_width, self.num_channels)
        self.assertEqual(image.shape, (self.img_height, self.img_width, self.num_channels))
        self.assertTrue(np.all(image.numpy() >= 0) and np.all(image.numpy() <= 1))

    @patch("ImageNormalization.normalize_image")
    def test_preprocess_image(self, mock_normalize):
        mock_normalize.return_value = tf.convert_to_tensor(np.random.rand(64, 64, 3), dtype=tf.float32)
        image = np.random.rand(64, 64, 3)
        processed_image = cd.preprocess_image(image, self.mean, self.std)
        self.assertEqual(processed_image.shape, (64, 64, 3))

    @patch("ConvertingData2.load_image")
    @patch("ConvertingData2.preprocess_image")
    def test_load_and_preprocess_image(self, mock_preprocess, mock_load):
        mock_load.return_value = np.random.rand(64, 64, 3)
        mock_preprocess.return_value = np.random.rand(64, 64, 3)
        result = cd.load_and_preprocess_image(self.image_path, self.img_height, self.img_width, self.num_channels, self.mean, self.std)
        self.assertEqual(result.shape, (64, 64, 3))

    @patch("numpy.savez_compressed")
    @patch("os.makedirs")
    def test_save_preprocessed_data(self, mock_makedirs, mock_savez):
        cd.save_preprocessed_data(self.folder, self.filename, self.data_key, self.data)
        mock_makedirs.assert_called_with(self.folder, exist_ok=True)
        mock_savez.assert_called()

    @patch("os.listdir", return_value=["img1.jpg", "img2.jpg", "img3.jpg"])
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps((np.array([0.5]), np.array([0.2]))))
    @patch("ConvertingData2.load_and_preprocess_image", side_effect=lambda *args: np.random.rand(64, 64, 3))
    @patch("ConvertingData2.save_preprocessed_data")
    def test_preprocess_dataset(self, mock_save, mock_load, mock_open, mock_listdir):
        cd.preprocess_dataset("input", "output", "processed_input", "processed_output", "mean_std.pkl", 64, 64, 3)
        mock_save.assert_called()

if __name__ == "__main__":
    unittest.main()
