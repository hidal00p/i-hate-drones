import argparse
from typing import NamedTuple

from bee_rl.enums import Algorithm


class TrainingArgs(NamedTuple):
    n_cpus: int
    av_ep_len: int
    n_episodes: int
    algo: Algorithm


def parse_training_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(
        description="Configure and run an attempt to synthesize an"
        " RL-based quadrocopter controller."
    )
    parser.add_argument("--n_cpus", type=int, default=2)
    parser.add_argument("--av_ep_len", type=int, default=5_000)
    parser.add_argument("--n_episodes", type=int, default=250)
    parser.add_argument(
        "--algo",
        type=Algorithm,
        default=Algorithm.SAC,
        choices=list(Algorithm),
    )
    return TrainingArgs(**vars(parser.parse_args()))
