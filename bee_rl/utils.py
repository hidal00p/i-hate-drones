import os
import logging
import logging.handlers
import time
import pathlib

LOGGER_NAME = os.getenv("LOGGER_NAME", "telemetry_logger")


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


def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > 0.04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


def configure_telemetry_logger():
    log_file = os.getenv("LOG_FILE", "telem.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file, mode="w", maxBytes=30_000_000, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
