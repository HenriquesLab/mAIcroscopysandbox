from typing import Optional
import numpy as np
import warnings

from .fluorescence_sim import generate_image
from .samples.sample import Sample

from matplotlib import pyplot as plt


class mAIcroscopySandbox(object):
    """
    Simulated fluorescence microscope with realistic imaging parameters.

    Provides a virtual microscopy environment with stage control, laser
    illumination, photobleaching, and detector noise characteristics.

    Parameters
    ----------
    stage_size : np.array, default=[5000, 5000]
        Size of the microscope stage in pixels [height, width].
    fov_size : np.array, default=[300, 300]
        Field of view size in pixels [height, width].
    laser_intensity : float, default=100000
        Maximum laser intensity in photons per pixel.
    pixel_size : float, default=100
        Physical size of one pixel in nanometers.

    Attributes
    ----------
    current_position : list
        Current stage position [row, col].
    laser_power : float
        Current laser power as percentage (0-100).
    sample : Sample
        Currently loaded sample object.
    """

    def __init__(
        self,
        stage_size: np.array = [5000, 5000],
        fov_size: np.array = [300, 300],
        laser_intensity: float = 100000,
        pixel_size: float = 100,
        sigma: float = 1.0,
        sigma_std: float = 0.01,
        gaussian_sigma: float = 2.0,
        output_dtype: str = "int16",
        random_seed: Optional[int] = None,
    ):
        self.stage_size = stage_size
        self.bleaching = np.ones(stage_size).astype(np.float32)
        self.fov_size = fov_size
        self.current_position = [self.fov_size[0] // 2, self.fov_size[1] // 2]
        self.laser_intensity = laser_intensity
        self.laser_power = 100
        self.wavelenght = 600
        self.wavelenght_std = 50
        self.NA = 1.2
        self.sigma = sigma
        self.sigma_std = sigma_std
        self.ADC_per_photon_conversion = 1.0
        self.ADC_offset = 100.0
        self.readout_noise = 50.0
        self.gaussian_sigma = gaussian_sigma
        self.output_dtype = output_dtype
        self.random_seed = random_seed

    def load_sample(self, sample: Sample, acquire: bool = False):
        """
        Load a sample onto the microscope stage.

        Parameters
        ----------
        sample : Sample
            Sample object to load.
        acquire : bool, default=False
            If True, acquire an image immediately after loading.

        Returns
        -------
        np.ndarray or None
            Acquired image if acquire=True, otherwise None.
        """

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        print(f"Loading sample of size: {sample.sample_size}")
        self.bleaching = np.ones(sample.sample_size).astype(np.float32)

        print("Resetting stage position to center position")
        self.current_position = [
            self.stage_size[0] // 2,
            self.stage_size[1] // 2,
        ]
        self.sample = sample

        if acquire:
            print("Acquiring image...")
            return self.acquire_image()

    def move_stage(self, movement: np.array = [0, 0], acquire: bool = False):
        """
        Move the microscope stage by specified offset.

        Parameters
        ----------
        movement : np.array, default=[0, 0]
            Movement offset in pixels [row_delta, col_delta].
        acquire : bool, default=False
            If True, acquire an image after moving.

        Returns
        -------
        np.ndarray or None
            Acquired image if acquire=True, otherwise None.
        """
        movement = self._check_move_stage(movement)
        self.current_position[0] += movement[0]
        self.current_position[1] += movement[1]

        if acquire:
            print(
                f"Stage Moved to position: {self.current_position}. Acquiring image..."
            )
            return self.acquire_image()
        else:
            print(f"Stage Moved to position: {self.current_position}")

    def set_laser_power(self, laser_power: float = 100.0):
        """
        Set the laser power percentage.

        Parameters
        ----------
        laser_power : float, default=100.0
            Laser power as percentage (0-100), clamped to valid range.
        """
        if laser_power > 100.0:
            laser_power = 100.0
        if laser_power < 0:
            laser_power = 0
        self.laser_power = laser_power

    def get_dtype(self, dtype_name: str = "int16"):
        """
        Get the numpy data type corresponding to the given name.

        Parameters
        ----------
        dtype_name : str, default="int16"
            Name of the desired data type.

        Returns
        -------
        np.dtype
            Corresponding numpy data type.
        """
        dtype_mapping = {
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
        }
        return dtype_mapping.get(dtype_name, np.int16)

    def acquire_image(self):
        """
        Acquire a fluorescence image at the current stage position.

        Generates an image with realistic photon noise, bleaching, detector
        noise, and Gaussian blur based on microscope parameters.

        Returns
        -------
        np.ndarray
            Simulated fluorescence image with noise and bleaching effects.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        sample_mask = self.sample.generate_mask()

        row_start = self.current_position[0] - self.fov_size[0] // 2
        row_end = self.current_position[0] + self.fov_size[0] // 2
        col_start = self.current_position[1] - self.fov_size[1] // 2
        col_end = self.current_position[1] + self.fov_size[1] // 2

        frame = generate_image(
            sample_mask[row_start:row_end, col_start:col_end],
            bleaching=self.bleaching[row_start:row_end, col_start:col_end],
            laser_intensity=(self.laser_power / 100) * self.laser_intensity,
            wavelenght=self.wavelenght,
            wavelenght_std=self.wavelenght_std,
            NA=self.NA,
            sigma=self.sigma,
            sigma_std=self.sigma_std,
            ADC_per_photon_conversion=self.ADC_per_photon_conversion,
            ADC_offset=self.ADC_offset,
            readout_noise=self.readout_noise,
            gaussian_sigma=self.gaussian_sigma,
        )

        bleaching_rate = self.sample.bleaching_rate
        self.bleaching[row_start:row_end, col_start:col_end] -= (
            self.bleaching[row_start:row_end, col_start:col_end]
            * bleaching_rate
            * (self.laser_power / 100)
        )
        self.bleaching[self.bleaching < 0] = 0

        return frame.astype(self.get_dtype(self.output_dtype))

    def set_wavelenght(self, wavelenght: float = 600.0):
        """Set excitation wavelength in nanometers."""
        self.wavelenght = wavelenght

    def set_wavelenght_std(self, wavelenght_std: float = 50.0):
        """Set standard deviation of wavelength in nanometers."""
        self.wavelenght_std = wavelenght_std

    def set_NA(self, NA: float = 1.2):
        """Set numerical aperture of the objective."""
        self.NA = NA

    def set_sigma(self, sigma: float = 0.21):
        """Set mean fluorophore emission standard deviation."""
        self.sigma = sigma

    def set_sigma_std(self, sigma_std: float = 0.01):
        """Set standard deviation of fluorophore emission."""
        self.sigma_std = sigma_std

    def set_ADC_per_photon_conversion(
        self, ADC_per_photon_conversion: float = 1.0
    ):
        """Set analog-to-digital conversion factor (ADC units per photon)."""
        self.ADC_per_photon_conversion = ADC_per_photon_conversion

    def set_ADC_offset(self, ADC_offset: float = 100.0):
        """Set baseline ADC offset value."""
        self.ADC_offset = ADC_offset

    def set_readout_noise(self, readout_noise: float = 50.0):
        """Set camera readout noise standard deviation in ADC units."""
        self.readout_noise = readout_noise

    def set_gaussian_sigma(self, gaussian_sigma: float = 5.0):
        """Set Gaussian blur sigma for point spread function."""
        self.gaussian_sigma = gaussian_sigma

    def _check_sample(self):
        if self.sample is None:
            raise ValueError("No sample loaded")

    def _check_move_stage(self, movement: np.array = [0, 0]):

        new_movement = movement

        if self.current_position[0] + movement[0] > self.stage_size[0]:
            warnings.warn(
                "Stage out of bounds, moving to furthest Y axis edge"
            )
            new_movement[0] = self.stage_size[0] - self.current_position[0]

        if self.current_position[1] + movement[1] > self.stage_size[1]:
            warnings.warn(
                "Stage out of bounds, moving to furthest X axis edge"
            )
            new_movement[1] = self.stage_size[1] - self.current_position[1]

        if self.current_position[0] + movement[0] < 0:
            warnings.warn(
                "Stage out of bounds, moving to furthest Y axis edge"
            )
            new_movement[0] = -self.current_position[0]

        if self.current_position[1] + movement[1] < 0:
            warnings.warn(
                "Stage out of bounds, moving to furthest X axis edge"
            )
            new_movement[1] = -self.current_position[1]

        return new_movement
