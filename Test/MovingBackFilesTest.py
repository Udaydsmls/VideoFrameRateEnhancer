import os
import shutil
import unittest
from FolderOperations import MovingBackFiles as mf


class TestFolderMerge(unittest.TestCase):
    def setUp(self):
        """Create test directory for storing all test files and folders."""
        self.test_root = os.path.join(os.getcwd(), "TestFilesFolders")
        os.makedirs(self.test_root, exist_ok=True)

        self.source_dir1 = os.path.join(self.test_root, "source1")
        self.source_dir2 = os.path.join(self.test_root, "source2")
        self.destination_dir = os.path.join(self.test_root, "destination")

        os.makedirs(self.source_dir1, exist_ok=True)
        os.makedirs(self.source_dir2, exist_ok=True)
        os.makedirs(self.destination_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test subdirectories without deleting the root Test directory."""
        shutil.rmtree(self.test_root)

    def test_merge_two_folders(self):
        """Test merging two folders with duplicate file handling."""

        # Create sample files in source_dir1
        with open(os.path.join(self.source_dir1, "file1.txt"), "w") as f:
            f.write("File 1 content")
        with open(os.path.join(self.source_dir1, "file2.txt"), "w") as f:
            f.write("File 2 content")

        # Create sample files in source_dir2
        with open(os.path.join(self.source_dir2, "file2.txt"), "w") as f:
            f.write("Different content for file 2")
        with open(os.path.join(self.source_dir2, "file3.txt"), "w") as f:
            f.write("File 3 content")

        mf.merge_folders(self.source_dir1, self.source_dir2, self.destination_dir)

        # Check if all expected files exist in the destination
        expected_files = {"file1.txt", "file2.txt", "file2_1.txt", "file3.txt"}
        actual_files = set(os.listdir(self.destination_dir))
        self.assertSetEqual(actual_files, expected_files)

    def test_move_back_files(self):
        """Test moving files back into structured subdirectories."""
        os.makedirs(os.path.join(self.source_dir1, "subdir"), exist_ok=True)
        os.makedirs(os.path.join(self.source_dir2, "subdir"), exist_ok=True)

        with open(os.path.join(self.source_dir1, "subdir", "fileA.txt"), "w") as f:
            f.write("Content A")
        with open(os.path.join(self.source_dir2, "subdir", "fileA.txt"), "w") as f:
            f.write("Different content A")

        mf.merge_subdirectories(self.source_dir1, self.source_dir2, self.destination_dir)

        expected_files = {"fileA.txt", "fileA_1.txt"}
        actual_files = set(os.listdir(os.path.join(self.destination_dir, "subdir")))
        self.assertSetEqual(actual_files, expected_files)


if __name__ == "__main__":
    unittest.main()
