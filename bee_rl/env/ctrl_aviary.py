from typing import Optional

import numpy as np
import pybullet as p
from gymnasium import spaces

from . import BaseAviary
from bee_rl.enums import DroneModel, Physics
from bee_rl.arena_elements import ElementGenerator
from bee_rl.utils import get_assets_dir


class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 1000,  # 1ms
        ctrl_freq: int = 200,  # 5ms
        obstacle_generator: Optional[ElementGenerator] = None,
        target_generator: Optional[ElementGenerator] = None,
        gui=False,
        record=False,
        user_debug_gui=True,
        output_folder="results",
    ):
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=False,
            user_debug_gui=user_debug_gui,
            output_folder=output_folder,
        )
        self.obstacle_generator = obstacle_generator
        self.target_generator = target_generator

    @property
    def _action_space(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array(
            [[0.0, 0.0, 0.0, 0.0] for _ in range(self.NUM_DRONES)]
        )
        act_upper_bound = np.array(
            [
                [self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM]
                for _ in range(self.NUM_DRONES)
            ]
        )
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    @property
    def _observation_space(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        # Observation vector
        #
        # Pos:                 X, Y, Z
        # Orientation quat:    Q1, Q2, Q3, Q4
        # Orientation eulr:    R, P, Y
        # Vel vector:          VX, VY, VZ
        # Angular vel vector:  WX, WY, WZ
        # Propeler rpm :       P0, P1, P2, P3
        obs_lower_bound = np.array(
            [
                [
                    -np.inf,
                    -np.inf,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -np.pi,
                    -np.pi,
                    -np.pi,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                for _ in range(self.NUM_DRONES)
            ]
        )
        obs_upper_bound = np.array(
            [
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.pi,
                    np.pi,
                    np.pi,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    self.MAX_RPM,
                    self.MAX_RPM,
                    self.MAX_RPM,
                    self.MAX_RPM,
                ]
                for _ in range(self.NUM_DRONES)
            ]
        )
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _compute_obs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array(
            [self._get_drone_state_vector(i) for i in range(self.NUM_DRONES)]
        )

    def _preprocess_action(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array(
            [np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)]
        )

    def _compute_reward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    def _compute_terminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    def _compute_truncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    def _compute_info(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years

    def _add_element_to_aviary(
        self, urdf_path: str, element_generator: ElementGenerator
    ) -> list[int]:
        z = 0.2
        engine_cli = self.CLIENT

        element_ids = [
            p.loadURDF(
                urdf_path,
                list(element_xy) + [z],
                p.getQuaternionFromEuler([0, 0, 0]),
                engine_cli,
            )
            for element_xy in element_generator.elements
        ]

        return element_ids

    def _add_obstacles(self):
        obstacle_urdf_path = str(get_assets_dir() / "column.urdf")
        self.obstacle_ids = self._add_element_to_aviary(
            obstacle_urdf_path, self.obstacle_generator
        )

    def _add_target(self):
        target_urdf_path = "duck_vhacd.urdf"
        self.target_id = self._add_element_to_aviary(
            target_urdf_path, self.target_generator
        )[0]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed, options)

        if self.obstacle_generator:
            self.obstacle_generator.reset()
            self._add_obstacles()

        if self.target_generator:
            self.target_generator.reset()
            self._add_target()

        return obs, info

    @property
    def orientation_plane(self):
        _, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0], self.CLIENT)
        orientationMatrix = p.getMatrixFromQuaternion(quat)

        xyz = np.vstack(
            (orientationMatrix[0:3], orientationMatrix[3:6], orientationMatrix[6:9])
        )
        return xyz
