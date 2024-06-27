import time
import logging
import numpy as np

from bee_rl.env import CtrlAviary
from bee_rl.control import PIDControl
from bee_rl.utils import sync, configure_telemetry_logger, LOGGER_NAME
from bee_rl.arena_elements import ArenaElementGenerator


def fly():
    Rxy = 0.5
    Z = 0.5
    Rz = 0.05 * Z
    phi_0 = 0.0
    theta_0 = 0.0
    phys_engine_freq_hz = 1000  # 1ms
    pid_freq_hz = 200  # 5 ms
    init_pos = np.array([np.cos(phi_0), np.sin(phi_0), np.sin(theta_0)]).reshape((1, 3))

    env = CtrlAviary(
        initial_xyzs=init_pos,
        pyb_freq=phys_engine_freq_hz,
        ctrl_freq=pid_freq_hz,
        gui=True,
        # obstacle_generator=ArenaElementGenerator((-1, 1), 10),
        # target_generator=ArenaElementGenerator((-1, 1), 1),
    )
    pid = PIDControl()

    start_time = time.time()
    action = np.zeros((1, 4))

    i = 0
    desired_speed = 0.1 * 3.6
    env.reset()

    logger = logging.getLogger(LOGGER_NAME)
    try:
        while True:

            # Step the simulation
            obs = env.step(action)[0]
            phi = i * np.pi / 2 / pid_freq_hz + phi_0
            theta = i * np.pi / pid_freq_hz
            target_p = np.hstack(
                [
                    [
                        Rxy * np.cos(phi),
                        Rxy * np.sin(phi),
                        Z + Rz * np.cos(theta),
                    ],
                ]
            )
            target_v = np.array(
                [
                    -0.6 * desired_speed * np.sin(phi),
                    0.6 * desired_speed * np.cos(phi),
                    -0.2 * desired_speed * np.sin(theta),
                ]
            )
            target_v /= np.linalg.norm(target_v)

            # Compute actuation commands given target position and velociity
            action[0, :] = pid.compute_control_from_state(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=target_p,
                target_vel=desired_speed * target_v,
            )[0]

            env.render()

            # Sync the simulation
            sync(i, start_time, env.CTRL_TIMESTEP)
            i += 1

            xyz = obs[0, 0:3]
            logger.info(f"{xyz[0]} {xyz[1]} {xyz[2]}")
    except KeyboardInterrupt:
        pass
    finally:
        # Close the environment
        env.close()


if __name__ == "__main__":
    configure_telemetry_logger()
    fly()
