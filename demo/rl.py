from bee_rl.training_engine import TrainingEngine


def run():
    training_engine = TrainingEngine(n_cpus=2, total_timesteps=25_000)
    training_engine.train()


if __name__ == "__main__":
    run()
