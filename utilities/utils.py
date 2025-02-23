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


def load_mean_std_file() -> tuple:
    """
    Loads mean and std file and returns mean and std as a tuple.
    """
    paths = setup.get_paths()
    file = paths['mean_std_file']

    if not os.path.isfile(file):
        raise FileNotFoundError("No mean std file found.")

    with open(file, "rb") as f:
        mean, std = pickle.load(f)

    return mean, std
