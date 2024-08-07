import time
import numpy as np

from bee_rl.env import CtrlAviary
from bee_rl.control import PathFollower
from bee_rl.utils import sync
from bee_rl.arena_elements import PositionElementGeneretor
from bee_rl.eyes import Eyes, VisionSpec
from bee_rl.trajectory import Trajectory


def clear_console():
    print("\x1b[2J")
    print("\033[0;0H")


def see():
    phys_engine_freq_hz = 1000  # 1ms
    pid_freq_hz = 200  # 5ms
    init_pos = np.array([[1.0, 0.0, 0.0]])

    env = CtrlAviary(
        eyes=Eyes(VisionSpec(cutoff_distance_m=1.25)),
        initial_xyzs=init_pos,
        pyb_freq=phys_engine_freq_hz,
        ctrl_freq=pid_freq_hz,
        gui=True,
        obstacle_generator=PositionElementGeneretor(
            [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]
        ),
    )
    controller = PathFollower(
        Trajectory.get_rollercoaster(Rxy=1.5, Z=0.2),
        desired_speed_ms=0.5 * 3.6,
        ctrl_timestep=env.CTRL_TIMESTEP,
    )

    action = np.zeros((1, 4))  # RPM
    env.reset()

    start_time = time.time()
    i = 0
    try:
        while True:
            # Step the simulation
            obs = env.step(action)[0]

            # Compute actuation commands given target position and velociity
            action[0, :] = controller.compute_rpm_from_state(state=obs[0])

            env.render()
            # Sync the simulation
            sync(i, start_time, env.CTRL_TIMESTEP)

            if i % 25 == 0:
                clear_console()
                print(list(np.round(obs[0, 21:], 6)))
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        # Close the environment
        env.close()
    pass


if __name__ == "__main__":
    see()
