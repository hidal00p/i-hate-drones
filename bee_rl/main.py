from bee_rl.args import parse_training_args
from bee_rl.training_engine import TrainingEngine


if __name__ == "__main__":
    TrainingEngine(parse_training_args()).train()
