import time

from stable_baselines3 import SAC
from bee_rl.training_engine import make_env
from bee_rl.utils import sync


def verify():
    env = make_env(gui=True)()
    control = SAC.load("../SAC.model")
    obs = env.reset()[0]

    start_time = time.time()
    i = 0
    try:
        while True:
            action = control.predict(obs, deterministic=True)[0]
            obs, _, term = env.step(action)[0:3]
            env.render()
            sync(i, start_time, env.CTRL_TIMESTEP)
            if term:
                obs = env.reset()[0]
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    verify()
