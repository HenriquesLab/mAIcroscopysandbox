import numpy as np
from tifffile import imread


class BinarySample(object):

    def __init__(
        self,
        img_path: str,
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
        self.img_path = img_path
        self.generate_mask()

    def generate_mask(self):
        binary = imread(self.img_path)
        return binary > 0

    def calculate_dynamics(self):
        pass
