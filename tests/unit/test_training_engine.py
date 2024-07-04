import pathlib
from unittest.mock import MagicMock, patch

from bee_rl.training_engine import TrainingEngine
from bee_rl.args import TrainingArgs
from bee_rl.enums import Algorithm


class TestTrainingEngine:

    @patch.object(pathlib.Path, "mkdir", MagicMock())
    @patch.object(TrainingEngine, "_init_env", MagicMock())
    def test_persistence_path(self):
        training_engine = TrainingEngine(
            TrainingArgs(n_cpus=2, av_ep_len=5000, n_episodes=250, algo=Algorithm.PPO)
        )
        expected_default_path = pathlib.Path("training_results/PPO.model")
        assert training_engine.model_file == expected_default_path
