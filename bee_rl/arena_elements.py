import numpy as np


class ArenaElementGenerator:

    def __init__(self, region: tuple[float, float], n: int = 10):
        self.region = region
        self.n = n
        self.reset()

    def reset(self):
        low, high = self.region
        x = np.random.uniform(low, high, self.n)
        y = np.random.uniform(low, high, self.n)

        self.obstacles = np.array([x, y]).reshape(self.n, 2)
