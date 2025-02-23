import os
import setup
import pickle


def load_latest_model() -> str:
    """
    Loads the latest model from the available models based on creation time.
    """
    paths = setup.get_paths()
    model_dir = paths["models"]
    models = os.listdir(model_dir)

    if not models:
        raise FileNotFoundError("No models found in the directory.")

    latest_model = max(models, key=lambda model: os.path.getctime(os.path.join(model_dir, model)))
    return os.path.join(model_dir, latest_model)


def load_dataset_dimensions() -> tuple:
    """Loads dataset dimensions from the first available file."""
    paths = setup.get_paths()
    dataset_dims_path = paths['dataset_dimensions']
    file_list = os.listdir(dataset_dims_path)

    if not file_list:
        raise FileNotFoundError("No dataset dimension files found.")

    with open(os.path.join(dataset_dims_path, file_list[0]), 'rb') as f:
        return pickle.load(f)
