import pathlib
from unittest.mock import MagicMock, patch

from bee_rl.training_engine import TrainingEngine, Algorithm


class TestTrainingEngine:

    @patch.object(pathlib.Path, "mkdir", MagicMock())
    @patch.object(TrainingEngine, "_init_env", MagicMock())
    def test_persistence_path(self):
        training_engine = TrainingEngine(algorithm=Algorithm.PPO)
        expected_default_path = pathlib.Path("training_results/PPO.model")
        assert training_engine.persistence_path == expected_default_path
