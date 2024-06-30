import numpy as np
from .pid import PIDControl


class PathFollower(PIDControl):

    def __init__(
        self, trajectory: np.ndarray, desired_speed_ms: float, ctrl_timestep: float
    ):
        super().__init__()
        self.trajectory = trajectory
        self.desired_speed = desired_speed_ms
        self.ctrl_timestep = ctrl_timestep

    def compute_rpm_from_state(self, state: np.ndarray) -> np.ndarray:
        our_location = state[0:3]
        target_location = self.compute_target_location(our_location)

        target_v = target_location - our_location
        target_v = self.desired_speed * target_v / np.linalg.norm(target_v)

        return self.compute_control_from_state(
            control_timestep=self.ctrl_timestep,
            state=state,
            target_pos=our_location,
            target_vel=target_v,
        )[0]

    def compute_target_location(self, our_location: np.ndarray) -> np.ndarray:
        trajectory_size = self.trajectory.shape[0]
        # This computation should be done smarter
        # e.g. take curvature + rate of change into account
        look_ahead_idx = int(0.05 * trajectory_size)

        delta_coord = self.trajectory - our_location
        distance = (delta_coord * delta_coord).sum(axis=1)
        new_target_idx = (np.argmin(distance) + look_ahead_idx) % trajectory_size

        return self.trajectory[new_target_idx]
