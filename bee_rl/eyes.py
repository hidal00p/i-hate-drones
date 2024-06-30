from typing import NamedTuple

import numpy as np
import pybullet as p


class VisionSpec(NamedTuple):
    """
    Params:
        angle: float              - observation angle
        segment_count: int        - number of segments used to discretize the angle
        cutoff_distance_m: float  - we do not observe beyond this distance
    """

    angle: float = 360.0 * np.pi / 180.0
    segment_count: int = 20
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
        for idx, (obstacle_id, _, _, hit_pos, _) in enumerate(collision_measurement):
            if obstacle_id == -1:
                continue
            self.observation[idx] = self.vision_spec.cutoff_distance_m - np.linalg.norm(
                np.array(hit_pos) - our_pos
            )

        return self.observation

    @property
    def observation_space(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bound = np.zeros(self.vision_spec.segment_count)
        upper_bound = np.full(
            self.vision_spec.segment_count, self.vision_spec.cutoff_distance_m
        )

        return lower_bound, upper_bound
