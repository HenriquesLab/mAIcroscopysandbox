import numpy as np


class Sample(object):

    def __init__(
        self,
        sample_size: np.array = [1000, 1000],
        bleaching_rate: float = 0.001,
        movement_rate: float = 10.0,
        movement_probability: float = 0.1,
    ):
        self.sample_size = sample_size
        self.movement_probability = movement_probability
        self.bleaching_rate = bleaching_rate
        self.movement_rate = movement_rate
        self.morphological_params = None
        self.generate_mask()

    def generate_mask(self):
        np.random.seed(0)
        return np.random.random(self.sample_size).astype(np.float32) > 0.8

    def calculate_dynamics(self):
        pass
