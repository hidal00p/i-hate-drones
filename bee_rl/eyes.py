from typing import NamedTuple

import numpy as np
import pybullet as p
from gymnasium import spaces


class VisionSpec(NamedTuple):
    """
    Params:
        angle: float              - observation angle
        segment_count: int        - number of segments used to discretize the angle
        clarity_drop_per_m: float - weight used to deminish the observation far away
        cutoff_distance_m: float  - we do not observe beyond this distance
    """

    angle: float = 360.0 * np.pi / 180.0
    segment_count: int = 20
    clarity_drop_per_m: float = 0.1
    cutoff_distance_m: float = 1.0


class Eyes:

    def __init__(self, vision_spec: VisionSpec = VisionSpec()):
        self.vision_spec = vision_spec
        self.setup_eyes()

    def setup_eyes(self):
        """
        Computes rays in the local planar coordinate system, which will be used
        for collision detection.
        """
        half_angle = self.vision_spec.angle / 2
        vision_bins = np.linspace(
            -half_angle, half_angle, self.vision_spec.segment_count + 1
        )
        ray_trace_angles = (vision_bins[1:] + vision_bins[:-1]) / 2

        cutoff_rad = self.vision_spec.cutoff_distance_m
        self.ray_vectors = np.vstack(
            (
                cutoff_rad * np.cos(ray_trace_angles),
                cutoff_rad * np.sin(ray_trace_angles),
            )
        ).T

        self.observation = np.zeros(self.ray_vectors.shape[0])

    def blink(self, our_pos: np.ndarray, local_coords: np.ndarray) -> np.ndarray:
        ray_vectors_global = np.matmul(self.ray_vectors, local_coords) + our_pos

        collision_measurement = p.rayTestBatch(
            rayFromPositions=[our_pos] * self.vision_spec.segment_count,
            rayToPositions=ray_vectors_global,
        )

        self.observation[:] = 0
        for idx, (obstacleId, _, _, hitPos, _) in enumerate(collision_measurement):
            if obstacleId == -1:
                continue
            self.observation[idx] = np.linalg.norm(np.array(list(hitPos)) - our_pos)

        return self.observation

    @property
    def observation_space(self) -> spaces.Box:
        minObservationVector = np.full((self.vision_spec.segment_count,), 0.0)
        maxObservationVector = np.full(
            (self.vision_spec.segment_count,), self.vision_spec.cutoff_distance_m
        )

        return spaces.Box(
            low=minObservationVector, high=maxObservationVector, dtype=np.float32
        )
