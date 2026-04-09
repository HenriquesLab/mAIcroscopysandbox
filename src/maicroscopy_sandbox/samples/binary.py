import numpy as np
from tifffile import imread


class BinarySample(object):
    """Sample backed by a binary TIFF image.

    Args:
        img_path: Path to the binary image on disk.
        sample_size: Sample dimensions in pixels as ``[height, width]``.
        bleaching_rate: Per-frame bleaching rate.
        movement_rate: Characteristic movement distance in pixels.
        movement_probability: Probability of movement per frame.
    """

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
        """Load the image and convert it to a binary mask.

        Returns:
            A boolean mask derived from the TIFF image.
        """
        binary = imread(self.img_path)
        return binary > 0

    def calculate_dynamics(self):
        """Binary samples are static, so there is no per-frame dynamics."""
        pass
