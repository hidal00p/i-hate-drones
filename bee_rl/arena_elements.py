from abc import abstractmethod
from typing import NamedTuple
import numpy as np


class ElementGenerator:

    _elements: np.ndarray

    @abstractmethod
    def reset(self):
        pass

    @property
    def elements(self) -> np.ndarray:
        return self._elements


class RandomElementGenerator(ElementGenerator):
    def __init__(self, region: tuple[float, float], n: int = 10):
        self.region = region
        self.n = n
        self.reset()

    def reset(self):
        low, high = self.region
        x = np.random.uniform(low, high, self.n)
        y = np.random.uniform(low, high, self.n)

        self._elements = np.array([x, y]).reshape(self.n, 2)


class _Position(NamedTuple):
    x: float
    y: float


class PositionElementGeneretor(ElementGenerator):
    def __init__(self, positions: list[_Position]):
        self._elements = np.zeros((len(positions), 2))
        for idx, (x, y) in enumerate(positions):
            self._elements[idx][0] = x
            self._elements[idx][1] = y

    def reset(self):
        pass
