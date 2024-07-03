import numpy as np

from . import CtrlAviary
from bee_rl.control import PathFollower


class RLAviary(CtrlAviary):
    @property
    def trajectory(self) -> np.ndarray:
        assert self._trajectory is not None
        return self._trajectory

    @trajectory.setter
    def trajectory(self, traj: np.ndarray):
        self._trajectory = traj

    @property
    def controller(self) -> PathFollower:
        return self._controller

    @controller.setter
    def controller(self, controller: PathFollower):
        self._controller = controller

    def step(self, rpm_correction):
        controller_rpm = self.controller.compute_rpm_from_state(
            self._get_drone_state_vector(0)
        )
        return super().step(controller_rpm + rpm_correction)

    def reset(self, *args, **kwargs):
        self.INIT_XYZS[0, :] = np.random.random(3) * 1.5
        return super().reset(*args, **kwargs)

    def _compute_reward(self):
        obs: np.ndarray = self._compute_obs()[0]

        # penalty for proximity to obstacles
        eye_saturation_penalty = (
            -1
            * (obs[21:]).sum()
            / self.eyes.vision_spec.cutoff_distance_m
            / self.eyes.vision_spec.segment_count
        )

        # penalty for being off-trajectory
        deviation: np.ndarray = self.trajectory - obs[:3]
        min_deviation = np.min((deviation * deviation).sum(axis=1))
        off_traj_penalty = np.exp(-min_deviation) - 3.0

        # collision penalty
        collision_penalty = (
            -5 * (obs[21:] > self.eyes.vision_spec.cutoff_distance_m - 0.1).any()
        )

        # penalty for having 0 altitude i.e. z = 0
        zero_alt_penalty = -5 * (obs[2] < 0.07)

        return (
            eye_saturation_penalty
            + off_traj_penalty
            + collision_penalty
            + zero_alt_penalty
        )
