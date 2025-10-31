import numpy as np


class Sample(object):
    """
    Base class for microscopy sample simulation.

    Provides basic structure for samples with fluorescence, bleaching,
    and movement dynamics.

    Parameters
    ----------
    sample_size : np.array, default=[1000, 1000]
        Size of the sample area in pixels [height, width].
    bleaching_rate : float, default=0.001
        Rate of fluorophore bleaching per frame.
    movement_rate : float, default=10.0
        Characteristic distance of fluorophore movement.
    movement_probability : float, default=0.1
        Probability of fluorophore movement per frame.

    Attributes
    ----------
    morphological_params : dict or None
        Optional parameters for sample morphology.
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
        """
        Generate fluorescence mask for the sample.

        Returns
        -------
        np.ndarray
            Binary mask with random fluorophore distribution.
        """
        np.random.seed(0)
        return np.random.random(self.sample_size).astype(np.float32) > 0.8

    def calculate_dynamics(self):
        """
        Update sample dynamics (movement, bleaching, etc.).

        To be implemented by subclasses.
        """
        pass
