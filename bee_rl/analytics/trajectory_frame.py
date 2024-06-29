import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class TrajectoryFrame:

    def __init__(self, traj_log_path: pathlib.Path | str):
        with open(traj_log_path, "r") as traj_file:
            trajectory_buffer = []
            for point_log in traj_file:
                # Just process a fixed format
                xyz = [float(coord) for coord in point_log[:-1].split(" ")]
                trajectory_buffer.append(xyz)

            self.trajectory = np.array(trajectory_buffer)

    def plot(self) -> tuple[Figure, Axes3D]:
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(projection="3d")

        x, y, z = self.trajectory.T[0], self.trajectory.T[1], self.trajectory.T[2]
        ax.plot(x, y, z)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig, ax
