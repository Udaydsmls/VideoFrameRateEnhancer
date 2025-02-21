import unittest
import numpy as np
from unittest.mock import patch
from PIL import Image
import ImageOperations.ImageNormalization as im


class TestImageProcessing(unittest.TestCase):

    @patch("ImageNormalization.Image.open")
    def test_compute_dataset_mean_std(self, mock_open):
        # Mocking image data
        img_data = np.random.rand(100, 100, 3).astype(np.float32)
        img_pil = Image.fromarray((img_data * 255).astype(np.uint8))

        mock_open.return_value = img_pil

        image_paths = ["fake_path_1.jpg", "fake_path_2.jpg"]
        mean, std = im.compute_dataset_mean_std(image_paths)

        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertTrue(np.all(std >= 0))  # Standard deviation should not be negative

    def test_normalize_image(self):
        image = np.random.rand(100, 100, 3).astype(np.float32)
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.2, 0.2, 0.2])

        normalized_image = im.normalize_image(image, mean, std)

        self.assertEqual(normalized_image.shape, image.shape)
        self.assertAlmostEqual(np.mean(normalized_image), np.mean((image - mean) / std), places=5)

    def test_denormalize_image(self):
        normalized_image = np.random.randn(100, 100, 3).astype(np.float32)
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.2, 0.2, 0.2])

        denormalized_image = im.denormalize_image(normalized_image, mean, std)

        self.assertEqual(denormalized_image.shape, normalized_image.shape)
        self.assertTrue(np.all(denormalized_image >= 0))  # Clipped to 0
        self.assertTrue(np.all(denormalized_image <= 1))  # Clipped to 1


if __name__ == "__main__":
    unittest.main()
