import pathlib

import matplotlib.pyplot as plt

from bee_rl.analytics.trajectory_frame import TrajectoryFrame
from bee_rl.trajectory import Trajectory


def visualize():
    traj_file_path = pathlib.Path("/Users/anton/home/rl/i-hate-drones/telem.log")
    assert traj_file_path.exists()

    _, graph = TrajectoryFrame(traj_file_path).plot()

    ref_trajectory = Trajectory.get_rollercoaster(Rxy=1.5, Rz=0.1, n_z=3)
    graph.plot(ref_trajectory.T[0], ref_trajectory.T[1], ref_trajectory.T[2])

    plt.show()


if __name__ == "__main__":
    visualize()
