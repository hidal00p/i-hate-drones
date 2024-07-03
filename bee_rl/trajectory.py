import numpy as np

from bee_rl.utils import NameSpace


class Trajectory(NameSpace):
    @classmethod
    def get_rollercoaster(
        _,
        Rxy: float = 1.0,
        Rz: float = 0.05,
        n_z: int = 2,
        Z: float = 0.5,
        grid_density: int = 1000,
    ) -> np.ndarray:
        assert isinstance(n_z, int), "Wave in Z direction should be a standing wave."

        phi = np.linspace(0, 2 * np.pi, grid_density)
        theta = n_z * phi

        x = Rxy * np.cos(phi)
        y = Rxy * np.sin(phi)
        z = Rz * np.sin(theta) + Z

        return np.vstack((x, y, z)).T
