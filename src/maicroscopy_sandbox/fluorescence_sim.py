import math
import numpy as np

from numba import njit
from skimage.filters import gaussian


def generate_image(
    mask: np.array,
    bleaching: np.array,
    laser_intensity: float = 1000.0,
    wavelenght: float = 600,
    wavelenght_std: float = 50,
    NA: float = 1.2,
    sigma: float = 0.21,
    sigma_std: float = 0.01,
    ADC_per_photon_conversion: float = 1.0,
    ADC_offset: float = 0.0,
    readout_noise: float = 50.0,
    gaussian_sigma: float = 5.0,
):
    """Generate a simulated fluorescence frame from a binary mask.

    Args:
        mask: Binary mask of active fluorophores.
        bleaching: Per-pixel bleaching map for the same region.
        laser_intensity: Maximum laser intensity in photons per pixel.
        wavelenght: Excitation wavelength in nanometers.
        wavelenght_std: Excitation wavelength standard deviation.
        NA: Objective numerical aperture.
        sigma: Mean fluorophore emission spread.
        sigma_std: Standard deviation of the emission spread.
        ADC_per_photon_conversion: Photon-to-ADU conversion factor.
        ADC_offset: Detector baseline offset.
        readout_noise: Detector noise standard deviation.
        gaussian_sigma: Gaussian blur applied at the end of the pipeline.

    Returns:
        A floating-point fluorescence image.
    """

    y_locs, x_locs = np.nonzero(mask)
    photon_array = np.random.normal(
        laser_intensity * 5, laser_intensity * 0.05, size=len(x_locs)
    )
    sigma = sigma * (wavelenght / 100) / NA
    sigma_std = sigma * (wavelenght_std / 100) / NA
    sigma_array = np.random.normal(sigma, sigma_std, size=len(x_locs))

    out = (
        FromLoc2Image_MultiThreaded(
            x_locs,
            y_locs,
            photon_array,
            sigma_array,
            mask.shape[0],
            mask.shape[1],
            1,
            mask,
            bleaching,
        )
        * mask
    )
    np.random.seed(None)
    out = (
        ADC_per_photon_conversion * np.random.poisson(out)
        + readout_noise * np.random.normal(size=(mask.shape[0], mask.shape[1]))
        + ADC_offset
    )
    out[out < 0] = 0
    out = gaussian(out, gaussian_sigma)
    return out


@njit
def FromLoc2Image_MultiThreaded(
    xc_array: np.ndarray,
    yc_array: np.ndarray,
    photon_array: np.ndarray,
    sigma_array: np.ndarray,
    image_height: int,
    image_width: int,
    pixel_size: float,
    mask: np.ndarray,
    bleaching: np.ndarray,
):
    """Accumulate emitter contributions into an image grid.

    Args:
        xc_array: X coordinates of emitters.
        yc_array: Y coordinates of emitters.
        photon_array: Photon counts for each emitter.
        sigma_array: Standard deviations for each emitter.
        image_height: Output image height in pixels.
        image_width: Output image width in pixels.
        pixel_size: Pixel size in the simulation grid.
        mask: Binary mask indicating pixels that should receive signal.
        bleaching: Per-pixel bleaching map.

    Returns:
        A 2D NumPy array with the simulated emitter contributions.
    """
    Image = np.zeros((image_height, image_width))
    half_pixel = pixel_size / 2.0

    for emitter_idx in range(len(xc_array)):
        xc = xc_array[emitter_idx]
        yc = yc_array[emitter_idx]
        photon = photon_array[emitter_idx]
        sigma = sigma_array[emitter_idx]

        # Don't bother if the emitter has photons <= 0 or if Sigma <= 0
        if (photon <= 0) or (sigma <= 0):
            continue

        S = sigma * math.sqrt(2.0)
        support_radius = 4.0 * sigma

        min_i = int(math.ceil((xc - support_radius - half_pixel) / pixel_size))
        max_i = int(math.floor((xc + support_radius - half_pixel) / pixel_size))
        min_j = int(math.ceil((yc - support_radius - half_pixel) / pixel_size))
        max_j = int(math.floor((yc + support_radius - half_pixel) / pixel_size))

        if min_i < 0:
            min_i = 0
        if min_j < 0:
            min_j = 0
        if max_i >= image_width:
            max_i = image_width - 1
        if max_j >= image_height:
            max_j = image_height - 1

        for j in range(min_j, max_j + 1):
            y = j * pixel_size - yc
            y_center = y + half_pixel
            for i in range(min_i, max_i + 1):
                if mask[j][i] <= 0:
                    continue

                x = i * pixel_size - xc
                x_center = x + half_pixel

                # Preserve the original distance gate exactly.
                if x_center * x_center + y_center * y_center < 16.0 * sigma**2:
                    ErfX = math.erf((x + pixel_size) / S) - math.erf(x / S)
                    ErfY = math.erf((y + pixel_size) / S) - math.erf(y / S)
                    Image[j][i] += (
                        0.25 * photon * ErfX * ErfY * bleaching[j][i]
                    )
    return Image


def binary2locs(img: np.ndarray, density: float):
    """Return a random subset of ``1`` pixels from a binary image.

    Args:
        img: Binary image containing candidate pixels.
        density: Fraction of active pixels to sample.

    Returns:
        Row and column index arrays for the selected pixels.
    """
    all_locs = np.nonzero(img == 1)
    n_points = int(len(all_locs[0]) * density)
    selected_idx = np.random.choice(len(all_locs[0]), n_points, replace=False)
    filtered_locs = all_locs[0][selected_idx], all_locs[1][selected_idx]
    return filtered_locs
