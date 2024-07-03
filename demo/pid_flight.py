import time
import logging
import numpy as np

from bee_rl.env import CtrlAviary
from bee_rl.control import PathFollower
from bee_rl.utils import sync, configure_telemetry_logger, LOGGER_NAME
from bee_rl.trajectory import Trajectory


def fly():
    gui = False
    phys_engine_freq_hz = 1000  # 1ms
    pid_freq_hz = 200  # 5ms
    init_pos = np.random.rand(1, 3) * 1.5
    init_pos[0, 2] = 0.0

    env = CtrlAviary(
        initial_xyzs=init_pos,
        pyb_freq=phys_engine_freq_hz,
        ctrl_freq=pid_freq_hz,
        gui=gui,
    )

    ref_trajectory = Trajectory.get_rollercoaster(Rxy=1.5, Rz=0.1, n_z=3)
    desired_speed_ms = 0.5 * 3.6
    controller = PathFollower(ref_trajectory, desired_speed_ms, env.CTRL_TIMESTEP)

    action = np.zeros((1, 4))  # RPM
    env.reset()

    start_time = time.time()
    logger = logging.getLogger(LOGGER_NAME)
    i = 0
    try:
        while True:
            # Step the simulation
            obs = env.step(action)[0]

            # Compute actuation commands given target position and velociity
            action[0, :] = controller.compute_rpm_from_state(
                state=obs[0],
            )

            if gui:
                env.render()
                # Sync the simulation
                sync(i, start_time, env.CTRL_TIMESTEP)

            xyz = obs[0, 0:3]
            logger.info(f"{xyz[0]} {xyz[1]} {xyz[2]}")
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        # Close the environment
        env.close()


if __name__ == "__main__":
    configure_telemetry_logger()
    fly()
