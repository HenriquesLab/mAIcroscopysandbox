from typing import Optional
import numpy as np
import warnings
from scipy.ndimage import gaussian_filter1d

from .fluorescence_sim import generate_image
from .samples.sample import Sample


class mAIcroscopySandbox(object):
    """Virtual fluorescence microscope with configurable optics and noise.

    Args:
        stage_size: Stage dimensions in pixels as ``[height, width]``.
        fov_size: Field-of-view dimensions in pixels as ``[height, width]``.
        laser_intensity: Maximum laser intensity in photons per pixel.
        pixel_size: Physical pixel size in nanometers.
        sigma: Mean fluorophore emission spread.
        sigma_std: Standard deviation of fluorophore emission spread.
        gaussian_sigma: Gaussian blur sigma for the point-spread function.
        output_dtype: Numpy dtype name used for the returned image.
        random_seed: Optional RNG seed for reproducible simulations.

    Attributes:
        current_position: Current stage position as ``[row, col]``.
        laser_power: Current laser power as a percentage.
        sample: Currently loaded sample instance.
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
        output_dtype: str = "uint16",
        random_seed: Optional[int] = None,
        voxel_size: tuple[float, float] | None = None,
        axial_binning_sigma: float | None = None,
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
        self.voxel_size = (
            (float(pixel_size), float(pixel_size))
            if voxel_size is None
            else (float(voxel_size[0]), float(voxel_size[1]))
        )
        self.axial_binning_sigma = axial_binning_sigma
        if random_seed is not None:
            np.random.seed(random_seed)

    def load_sample(self, sample: Sample, acquire: bool = False):
        """Load a sample onto the stage.

        Args:
            sample: Sample object to load.
            acquire: If ``True``, acquire an image immediately.

        Returns:
            The acquired frame when ``acquire`` is ``True``; otherwise ``None``.
        """

        print(f"Loading sample of size: {sample.sample_size}")
        bleaching_shape = getattr(sample, "volume_shape", sample.sample_size)
        self.bleaching = np.ones(bleaching_shape).astype(np.float32)

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
        """Move the stage by an offset in pixels.

        Args:
            movement: Offset as ``[row_delta, col_delta]``.
            acquire: If ``True``, acquire an image after moving.

        Returns:
            The acquired frame when ``acquire`` is ``True``; otherwise ``None``.
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
        """Set laser power as a percentage.

        Args:
            laser_power: Requested laser power percentage.
        """
        if laser_power > 100.0:
            laser_power = 100.0
        if laser_power < 0:
            laser_power = 0
        self.laser_power = laser_power

    def get_dtype(self, dtype_name: str = "uint16"):
        """Return the NumPy dtype matching ``dtype_name``.

        Args:
            dtype_name: String key such as ``"uint16"`` or ``"float32"``.

        Returns:
            The matching NumPy dtype, or ``np.uint16`` when unknown.
        """
        dtype_mapping = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
        }
        return dtype_mapping.get(dtype_name, np.uint16)

    def _to_output_dtype(self, frame: np.ndarray) -> np.ndarray:
        """Clip a float frame to the output dtype's representable range and cast.

        Prevents silent overflow/wraparound (e.g. large positive values wrapping
        to negative numbers when cast to a signed/unsigned integer dtype).

        Args:
            frame: Simulated frame as a floating-point array.

        Returns:
            The frame clipped and cast to ``self.output_dtype``.
        """
        dtype = self.get_dtype(self.output_dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            frame = np.clip(frame, info.min, info.max)
        return frame.astype(dtype)

    def acquire_image(self):
        """Acquire a fluorescence frame at the current stage position.

        Returns:
            The simulated frame converted to the configured output dtype.
        """
        sample_mask = self._generate_sample_signal()

        row_start = self.current_position[0] - self.fov_size[0] // 2
        row_end = self.current_position[0] + self.fov_size[0] // 2
        col_start = self.current_position[1] - self.fov_size[1] // 2
        col_end = self.current_position[1] + self.fov_size[1] // 2
        if sample_mask.ndim == 3:
            z_index = sample_mask.shape[0] // 2
            sample_mask = sample_mask[z_index]
            bleaching = self.bleaching[z_index]
        else:
            bleaching = self.bleaching

        frame = generate_image(
            sample_mask[row_start:row_end, col_start:col_end],
            bleaching=bleaching[row_start:row_end, col_start:col_end],
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
        bleaching_region = bleaching[row_start:row_end, col_start:col_end]
        bleaching_region -= bleaching_region * bleaching_rate * (self.laser_power / 100)
        self.bleaching[self.bleaching < 0] = 0

        return self._to_output_dtype(frame)

    def acquire_stack(self):
        """Acquire a fluorescence z stack from a 3D sample.

        Returns:
            A stack with shape ``(z, y, x)`` converted to the configured dtype.
        """
        sample_signal = self._generate_sample_signal()
        if sample_signal.ndim != 3:
            sample_signal = sample_signal[np.newaxis, :, :]

        row_start = self.current_position[0] - self.fov_size[0] // 2
        row_end = self.current_position[0] + self.fov_size[0] // 2
        col_start = self.current_position[1] - self.fov_size[1] // 2
        col_end = self.current_position[1] + self.fov_size[1] // 2

        cropped_signal = sample_signal[:, row_start:row_end, col_start:col_end]
        if self.bleaching.ndim == 3:
            cropped_bleaching = self.bleaching[:, row_start:row_end, col_start:col_end]
        else:
            cropped_bleaching = np.broadcast_to(
                self.bleaching[row_start:row_end, col_start:col_end],
                cropped_signal.shape,
            ).astype(np.float32)

        stack = np.empty(cropped_signal.shape, dtype=np.float32)
        for z_index in range(cropped_signal.shape[0]):
            stack[z_index] = generate_image(
                cropped_signal[z_index],
                bleaching=cropped_bleaching[z_index],
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
        if self.bleaching.ndim == 3:
            bleaching_region = self.bleaching[:, row_start:row_end, col_start:col_end]
            bleaching_region -= (
                bleaching_region * bleaching_rate * (self.laser_power / 100)
            )
        else:
            bleaching_region = self.bleaching[row_start:row_end, col_start:col_end]
            bleaching_region -= (
                bleaching_region * bleaching_rate * (self.laser_power / 100)
            )
        self.bleaching[self.bleaching < 0] = 0
        return self._to_output_dtype(stack)

    def _generate_sample_signal(self):
        if hasattr(self.sample, "generate_volume"):
            sample_signal = self.sample.generate_volume()
        else:
            sample_signal = self.sample.generate_mask()
        return self._bin_axial_signal(sample_signal)

    def _bin_axial_signal(self, sample_signal):
        if sample_signal.ndim != 3:
            return sample_signal

        axial_sigma = self.axial_binning_sigma
        if axial_sigma is None:
            axial_sigma = max(0.0, self.voxel_size[1] / self.voxel_size[0] - 1.0)
        if axial_sigma <= 0:
            return sample_signal
        return gaussian_filter1d(
            sample_signal.astype(np.float32),
            sigma=axial_sigma,
            axis=0,
            mode="nearest",
        )

    def set_wavelenght(self, wavelenght: float = 600.0):
        """Set the excitation wavelength in nanometers.

        Args:
            wavelenght: Excitation wavelength in nanometers.
        """
        self.wavelenght = wavelenght

    def set_wavelenght_std(self, wavelenght_std: float = 50.0):
        """Set the wavelength standard deviation in nanometers.

        Args:
            wavelenght_std: Standard deviation for excitation wavelength.
        """
        self.wavelenght_std = wavelenght_std

    def set_NA(self, NA: float = 1.2):
        """Set the objective numerical aperture.

        Args:
            NA: Objective numerical aperture.
        """
        self.NA = NA

    def set_sigma(self, sigma: float = 0.21):
        """Set the mean fluorophore emission spread.

        Args:
            sigma: Mean fluorophore emission spread.
        """
        self.sigma = sigma

    def set_sigma_std(self, sigma_std: float = 0.01):
        """Set the fluorophore emission spread standard deviation.

        Args:
            sigma_std: Standard deviation of the fluorophore spread.
        """
        self.sigma_std = sigma_std

    def set_ADC_per_photon_conversion(self, ADC_per_photon_conversion: float = 1.0):
        """Set the analog-to-digital conversion factor.

        Args:
            ADC_per_photon_conversion: Conversion factor from photons to ADU.
        """
        self.ADC_per_photon_conversion = ADC_per_photon_conversion

    def set_ADC_offset(self, ADC_offset: float = 100.0):
        """Set the detector baseline offset.

        Args:
            ADC_offset: Additive detector baseline in ADU.
        """
        self.ADC_offset = ADC_offset

    def set_readout_noise(self, readout_noise: float = 50.0):
        """Set the camera readout noise standard deviation.

        Args:
            readout_noise: Standard deviation of the detector noise.
        """
        self.readout_noise = readout_noise

    def set_gaussian_sigma(self, gaussian_sigma: float = 5.0):
        """Set the Gaussian blur sigma used for the point-spread function.

        Args:
            gaussian_sigma: Standard deviation of the PSF blur kernel.
        """
        self.gaussian_sigma = gaussian_sigma

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
