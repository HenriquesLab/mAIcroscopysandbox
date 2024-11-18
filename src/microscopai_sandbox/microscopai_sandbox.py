import numpy as np
import warnings

from .fluorescence_sim import generate_image
from .samples.sample import Sample

from matplotlib import pyplot as plt


class MicroscopAIsandbox(object):
    def __init__(
        self,
        stage_size: np.array = [5000, 5000],
        fov_size: np.array = [300, 300],
        laser_intensity: float = 100000,
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
        self.sigma = 1
        self.sigma_std = 0.01
        self.ADC_per_photon_conversion = 1.0
        self.ADC_offset = 100.0
        self.readout_noise = 50.0

    def load_sample(self, sample: Sample, acquire: bool = False):

        print(f"Loading sample of size: {sample.sample_size}")
        self.bleaching = np.ones(sample.sample_size).astype(np.float32)

        print("Resetting stage position to center position")
        self.current_position = [self.stage_size[0] // 2, self.stage_size[1] // 2]
        self.sample = sample

        if acquire:
            print("Acquiring image...")
            return self.acquire_image()

    def move_stage(self, movement: np.array = [0, 0], acquire: bool = False):
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
        if laser_power > 100.0:
            laser_power = 100.0
        if laser_power < 0:
            laser_power = 0
        self.laser_power = laser_power

    def acquire_image(self):

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
        )

        bleaching_rate = self.sample.bleaching_rate
        self.bleaching[row_start:row_end, col_start:col_end] -= bleaching_rate * (
            self.laser_power / 100
        )
        self.bleaching[self.bleaching < 0] = 0

        return frame

    def set_wavelenght(self, wavelenght: float = 600.0):
        self.wavelenght = wavelenght

    def set_wavelenght_std(self, wavelenght_std: float = 50.0):
        self.wavelenght_std = wavelenght_std

    def set_NA(self, NA: float = 1.2):
        self.NA = NA

    def set_sigma(self, sigma: float = 0.21):
        self.sigma = sigma

    def set_sigma_std(self, sigma_std: float = 0.01):
        self.sigma_std = sigma_std

    def set_ADC_per_photon_conversion(self, ADC_per_photon_conversion: float = 1.0):
        self.ADC_per_photon_conversion = ADC_per_photon_conversion

    def set_ADC_offset(self, ADC_offset: float = 100.0):
        self.ADC_offset = ADC_offset

    def set_readout_noise(self, readout_noise: float = 50.0):
        self.readout_noise = readout_noise

    def _check_sample(self):
        if self.sample is None:
            raise ValueError("No sample loaded")

    def _check_move_stage(self, movement: np.array = [0, 0]):

        new_movement = movement

        if self.current_position[0] + movement[0] > self.stage_size[0]:
            warnings.warn("Stage out of bounds, moving to furthest Y axis edge")
            new_movement[0] = self.stage_size[0] - self.current_position[0]

        if self.current_position[1] + movement[1] > self.stage_size[1]:
            warnings.warn("Stage out of bounds, moving to furthest X axis edge")
            new_movement[1] = self.stage_size[1] - self.current_position[1]

        if self.current_position[0] + movement[0] < 0:
            warnings.warn("Stage out of bounds, moving to furthest Y axis edge")
            new_movement[0] = -self.current_position[0]

        if self.current_position[1] + movement[1] < 0:
            warnings.warn("Stage out of bounds, moving to furthest X axis edge")
            new_movement[1] = -self.current_position[1]

        return new_movement
