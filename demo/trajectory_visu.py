import pathlib

import numpy as np
import matplotlib.pyplot as plt

from bee_rl.analytics.trajectory import TrajectoryFrame


def generate_ref_trajectory(
    density: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rxy = 0.5
    Z = 0.5
    Rz = 0.05 * Z
    phi = np.linspace(0, 2 * np.pi, density)
    theta = 2 * phi
    x, y, z = Rxy * np.cos(phi), Rxy * np.sin(phi), Z + Rz * np.sin(theta)
    return x, y, z


def visualize():
    traj_file_path = pathlib.Path("/Users/anton/home/rl/i-hate-drones/telem.log")
    assert traj_file_path.exists()

    _, graph = TrajectoryFrame(traj_file_path).plot()

    x_ref, y_ref, z_ref = generate_ref_trajectory()
    graph.plot(x_ref, y_ref, z_ref)

    plt.show()


if __name__ == "__main__":
    visualize()
