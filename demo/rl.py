import time

import numpy as np
from stable_baselines3 import SAC

from bee_rl.trajectory import Trajectory
from bee_rl.arena_elements import PositionElementGeneretor
from bee_rl.eyes import Eyes, VisionSpec
from bee_rl.control import PathFollower
from bee_rl.env import RLAviary
from bee_rl.utils import sync


def make_env(gui=False) -> RLAviary:
    env = RLAviary(
        initial_xyzs=np.array([[2.5, 2.5, 0.0]]),
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
        desired_speed_ms=0.4 * 3.6,
        ctrl_timestep=env.CTRL_TIMESTEP,
    )
    return env


def train():
    env = make_env()
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=250 * 10_000, log_interval=1)
    env.close()

    gui_env = make_env(True)
    obs = gui_env.reset()[0]

    start_time = time.time()
    i = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs = gui_env.step(action)[0]
            gui_env.render()
            sync(i, start_time, env.CTRL_TIMESTEP)
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        gui_env.close()


if __name__ == "__main__":
    train()
