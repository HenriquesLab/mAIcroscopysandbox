import numpy as np


class Sample(object):
    """Base class for synthetic microscopy samples.

    Args:
        sample_size: Sample dimensions in pixels as ``[height, width]``.
        bleaching_rate: Per-frame bleaching rate.
        movement_rate: Characteristic movement distance in pixels.
        movement_probability: Probability of movement per frame.

    Attributes:
        morphological_params: Optional morphology parameters for subclasses.
    """

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
        """Generate a default binary fluorophore mask.

        Returns:
            A ``float32`` binary mask with a deterministic random pattern.
        """
        np.random.seed(0)
        return np.random.random(self.sample_size).astype(np.float32) > 0.8

    def calculate_dynamics(self):
        """Update sample dynamics.

        Subclasses override this to advance their own state between frames.
        """
        pass
