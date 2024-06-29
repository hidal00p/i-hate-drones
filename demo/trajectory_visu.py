import pathlib

import numpy as np
import matplotlib.pyplot as plt

from bee_rl.analytics.trajectory_frame import TrajectoryFrame
from bee_rl.trajectory import Trajectory


def generate_ref_trajectory() -> np.ndarray:
    Rxy = 0.5
    Z = 0.5
    Rz = 0.1 * Z
    return Trajectory.get_rollercoaster(Rxy=Rxy, Rz=Rz, n_z=2, Z=Z)


def visualize():
    traj_file_path = pathlib.Path("/Users/anton/home/rl/i-hate-drones/telem.log")
    assert traj_file_path.exists()

    _, graph = TrajectoryFrame(traj_file_path).plot()

    ref_trajectory = generate_ref_trajectory()
    graph.plot(ref_trajectory.T[0], ref_trajectory.T[1], ref_trajectory.T[2])

    plt.show()


if __name__ == "__main__":
    visualize()
