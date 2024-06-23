import os
import pathlib


def add(a: float, b: float):
    return a + b


def get_assets_dir() -> pathlib.Path:
    """
    Retrieves the directory which contains the .urdf model
    of the drone's digital twin.
    """

    assets_dir = os.getenv("ASSETS_DIR")
    assert (
        assets_dir
    ), "Path to assets directory is not provided. Please define it as an environment variable before running the script."
    return pathlib.Path(assets_dir)
