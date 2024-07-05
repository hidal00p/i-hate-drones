import pathlib
from contextlib import contextmanager

import numpy as np
from stable_baselines3 import SAC, DDPG, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from bee_rl.trajectory import Trajectory
from bee_rl.arena_elements import PositionElementGeneretor
from bee_rl.eyes import Eyes, VisionSpec
from bee_rl.control import PathFollower
from bee_rl.env import RLAviary
from bee_rl.args import TrainingArgs
from bee_rl.enums import Algorithm


def make_env(gui=False):
    def _init() -> RLAviary:
        """
        Generates default RL env suitable for training.
        """
        env = RLAviary(
            initial_xyzs=np.random.random((1, 3)),
            eyes=Eyes(VisionSpec(angle=120 * np.pi / 180)),
            gui=gui,
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


class TrainingEngine:

    def __init__(self, training_args: TrainingArgs):
        self.training_args = training_args
        self._init_env()

    def _init_env(self):
        n_cpus = self.training_args.n_cpus
        self.vec_env = SubprocVecEnv([make_env()] * n_cpus)

        if self.training_args.algo == Algorithm.SAC:
            self.model = SAC("MlpPolicy", self.vec_env, verbose=1)
            return

        if self.training_args.algo == Algorithm.DDPG:
            self.model = DDPG("MlpPolicy", self.vec_env, verbose=1)
            return

        if self.training_args.algo == Algorithm.PPO:
            self.model = PPO("MlpPolicy", self.vec_env, verbose=1)
            return

        raise RuntimeError(f"{self.training_args.algo.value} is unsupported.")

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
            self.model.save(self.model_file)
            self.vec_env.close()

    def train(self):
        total_ts = self.training_args.av_ep_len * self.training_args.n_episodes
        save_freq = total_ts // self.training_args.n_cpus // 10
        with self.safe_operation():
            self.model.learn(
                total_timesteps=total_ts,
                log_interval=15,
                callback=_SaveModelCallback(save_freq=save_freq, training_engine=self),
            )

    @property
    def persistence_dir(self) -> pathlib.Path:
        directory = pathlib.Path("training_results")
        directory.mkdir(exist_ok=True)
        return directory

    @property
    def model_file(self) -> pathlib.Path:
        return self.persistence_dir / f"{self.training_args.algo.value}.model"

    @property
    def meta_file(self) -> pathlib.Path:
        return self.persistence_dir / f"{self.training_args.algo.value}.meta"


class _SaveModelCallback(BaseCallback):

    def __init__(self, save_freq: int, training_engine: TrainingEngine):
        self.training_engine = training_engine
        self.save_freq = save_freq
        super().__init__()

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_engine.model.save(self.training_engine.model_file)

        return True
