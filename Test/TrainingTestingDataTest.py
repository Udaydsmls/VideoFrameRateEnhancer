import os
import shutil
import unittest
import pickle
import FolderOperations.TrainingTestingData as ttd
from PIL import Image


def create_dummy_image(image_path):
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))  # Creates a white image
    img.save(image_path)


class TestImageSeparation(unittest.TestCase):

    def setUp(self):
        # Setup test directories inside 'Test'
        self.test_dir = 'temp_test'
        self.source_folder = os.path.join(self.test_dir, 'source')
        self.input_train_folder = os.path.join(self.test_dir, 'train')
        self.output_train_folder = os.path.join(self.test_dir, 'test')
        self.output_pickle_path = os.path.join(self.test_dir, 'stats.pkl')

        os.makedirs(self.source_folder, exist_ok=True)
        os.makedirs(self.input_train_folder, exist_ok=True)
        os.makedirs(self.output_train_folder, exist_ok=True)

    def tearDown(self):
        # Remove test directories
        shutil.rmtree(self.test_dir, ignore_errors=True)

    #
    def test_separate_images(self):
        # Create mock images
        for i in range(0, 5):
            with open(os.path.join(self.source_folder, f'frame_video_{i:06d}.jpg'), 'w') as f:
                f.write('test image content')
        ttd.distribute_images(self.source_folder, self.input_train_folder, self.output_train_folder)

        train_files = os.listdir(self.input_train_folder)
        test_files = os.listdir(self.output_train_folder)

        self.assertEqual(len(train_files), 3)  # Expecting 1, 3, 5 in train
        self.assertEqual(len(test_files), 2)  # Expecting 2, 4 in test

    def test_separate_frames(self):
        os.makedirs(os.path.join(self.source_folder, 'video1'), exist_ok=True)
        for i in range(0, 5):
            create_dummy_image(os.path.join(self.source_folder, 'video1', f'frame_video_{i:06d}.jpg'))

        ttd.process_image_directories(self.source_folder, self.input_train_folder, self.output_train_folder, self.output_pickle_path)

        train_files = os.listdir(os.path.join(self.input_train_folder, 'video1'))
        test_files = os.listdir(os.path.join(self.output_train_folder, 'video1'))

        self.assertEqual(len(train_files), 3)
        self.assertEqual(len(test_files), 2)
        self.assertTrue(os.path.exists(self.output_pickle_path))

        with open(self.output_pickle_path, 'rb') as f:
            data = pickle.load(f)
        self.assertEqual(len(data), 2)  # Expecting mean and std


if __name__ == '__main__':
    unittest.main()
