import os
import gc
import pickle
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
import CreatingModel.Model as ml
import setup


def configure_gpu_memory():
    """Configures TensorFlow to prevent memory allocation issues."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def check_dataset_dimensions() -> bool:
    """
    Checks if all dataset images have the same dimensions.
    """
    paths = setup.get_paths()
    dataset_dims_path = paths['dataset_dimensions']

    reference_dims = None

    for file in os.listdir(dataset_dims_path):
        with open(os.path.join(dataset_dims_path, file), 'rb') as f:
            current_dims = pickle.load(f)

        if reference_dims is None:
            reference_dims = current_dims
        elif current_dims != reference_dims:
            print("Dataset image dimensions mismatch.")
            return False

    return True


def load_dataset_dimensions() -> tuple:
    """Loads dataset dimensions from the first available file."""
    paths = setup.get_paths()
    dataset_dims_path = paths['dataset_dimensions']
    file_list = os.listdir(dataset_dims_path)

    if not file_list:
        raise FileNotFoundError("No dataset dimension files found.")

    with open(os.path.join(dataset_dims_path, file_list[0]), 'rb') as f:
        return pickle.load(f)


def train_model() -> None:
    """
    Trains the image translation model and saves it to the specified directory.
    """
    paths = setup.get_paths()
    params = setup.get_model_params()

    if not check_dataset_dimensions():
        return

    img_height, img_width, num_channels = load_dataset_dimensions()
    model = ml.create_image_translation_model(img_height, img_width, num_channels)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print(model.summary())

    for dataset_idx, dataset in enumerate(os.listdir(paths['input_training_dataset'])):
        input_path = os.path.join(paths['input_training_dataset'], dataset)
        output_path = os.path.join(paths['output_training_dataset'], dataset)

        train_files = sorted(
            [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".npz")]
        )

        test_files = sorted(
            [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".npz")]
        )

        for file_idx, (train_file, test_file) in enumerate(zip(train_files, test_files)):
            train_data = np.load(train_file)["input"]
            test_data = np.load(test_file)["output"]

            X1, X2 = map(np.array, zip(*train_data))
            Y = np.array(test_data)

            min_size = min(len(X1), len(Y))
            X1, X2, Y = X1[:min_size], X2[:min_size], Y[:min_size]

            X1, X2, Y = shuffle(X1, X2, Y, random_state=42)

            model.fit([X1, X2], Y, epochs=params['num_epochs'], batch_size=params['batch_size'],
                      validation_split=params['validation_split'])

            K.clear_session()
            gc.collect()

            model.save(os.path.join(paths['models'],
                        f"image_translation_model_{img_height}_{img_width}_{num_channels}_ver_{dataset_idx}_{file_idx}"),
                       save_format="tf")
