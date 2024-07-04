import pathlib
from contextlib import contextmanager
from enum import Enum

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from bee_rl.trajectory import Trajectory
from bee_rl.arena_elements import PositionElementGeneretor
from bee_rl.eyes import Eyes, VisionSpec
from bee_rl.control import PathFollower
from bee_rl.env import RLAviary


def make_env() -> RLAviary:
    def _init():
        """
        Generates default RL env suitable for training.
        """
        env = RLAviary(
            initial_xyzs=np.random.random((1, 3)),
            eyes=Eyes(VisionSpec(angle=120 * np.pi / 180)),
            gui=False,
            pyb_freq=1000,
            ctrl_freq=250,
            obstacle_generator=PositionElementGeneretor(
                positions=[(0.0, 1.5), (0.0, -1.5)]
            ),
        )

        # Inject a trajectory to follow
        env.trajectory = Trajectory.get_rollercoaster(Rxy=1.5, Z=0.2, Rz=0.0)

        # Inject a classical controller
        env.controller = PathFollower(
            trajectory=env.trajectory,
            desired_speed_ms=0.3 * 3.6,
            ctrl_timestep=env.CTRL_TIMESTEP,
        )

        env.reset()
        return env

    return _init


class Algorithm(Enum):
    SAC = "SAC"
    PPO = "PPO"
    DDPG = "DDPG"


class TrainingEngine:

    def __init__(
        self,
        n_cpus: int = 4,
        algorithm: Algorithm = Algorithm.SAC,
        total_timesteps: int = 2_500_000,  # ~250 episodes
    ):
        self.n_cpus = n_cpus
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self._init_env()

    def _init_env(self):
        self.vec_env = SubprocVecEnv([make_env()] * self.n_cpus)
        self.model = SAC("MlpPolicy", self.vec_env, verbose=1)

    @contextmanager
    def safe_operation(self):
        # TODO: we need a global logger
        try:
            yield
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        finally:
            self.model.save(self.persistence_path)
            self.vec_env.close()

    def train(self):
        with self.safe_operation():
            self.model.learn(total_timesteps=self.total_timesteps, log_interval=15)

    @property
    def persistence_path(self) -> pathlib.Path:
        directory = pathlib.Path("training_results")
        directory.mkdir(exist_ok=True)
        return directory / f"{self.algorithm.value}.model"
