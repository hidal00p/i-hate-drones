import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from bee_rl.trajectory import Trajectory
from bee_rl.arena_elements import PositionElementGeneretor
from bee_rl.eyes import Eyes, VisionSpec
from bee_rl.control import PathFollower
from bee_rl.env import RLAviary


def make_env(gui=False) -> RLAviary:
    def _init():
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
        env.trajectory = Trajectory.get_rollercoaster(Rxy=1.5, Z=0.2, Rz=0.0)
        env.controller = PathFollower(
            trajectory=env.trajectory,
            desired_speed_ms=0.25 * 3.6,
            ctrl_timestep=env.CTRL_TIMESTEP,
        )
        env.reset()
        return env

    return _init


def train():
    n_cpus = 4
    vec_env = SubprocVecEnv([make_env()] * n_cpus)
    model = SAC("MlpPolicy", vec_env, verbose=1)
    try:
        model.learn(total_timesteps=25 * 10_000, log_interval=5)
    except KeyboardInterrupt:
        pass
    finally:
        model.save("/Users/anton/home/rl/i-hate-drones/models/trial.model")
        vec_env.close()


if __name__ == "__main__":
    train()
