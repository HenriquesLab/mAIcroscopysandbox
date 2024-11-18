import math
import numpy as np

from numba import njit, prange


def generate_image(
    mask: np.array,
    bleaching: np.array,
    laser_intensity: float = 100000.0,
    wavelenght: float = 600,
    wavelenght_std: float = 50,
    NA: float = 1.2,
    sigma: float = 0.21,
    sigma_std: float = 0.01,
    ADC_per_photon_conversion: float = 1.0,
    ADC_offset: float = 100.0,
    readout_noise: float = 50.0,
):

    y_locs, x_locs = np.nonzero(mask)
    photon_array = np.random.normal(
        laser_intensity * 5, laser_intensity * 0.05, size=len(x_locs)
    )
    sigma = sigma * (wavelenght / 100) / NA
    sigma_std = (
        sigma * (wavelenght_std / 100) / NA
    )
    sigma_array = np.random.normal(sigma, sigma_std, size=len(x_locs))

    out = FromLoc2Image_MultiThreaded(
                x_locs,
                y_locs,
                photon_array,
                sigma_array,
                mask.shape[0],
                mask.shape[1],
                1,
                bleaching
            ) * mask
    out = (
        ADC_per_photon_conversion * np.random.poisson(out)
        + readout_noise
        * np.random.normal(size=(mask.shape[0], mask.shape[1]))
        + ADC_offset
    )
    out[out < 0] = 0
    return out


@njit(parallel=True)
def FromLoc2Image_MultiThreaded(
    xc_array: np.ndarray,
    yc_array: np.ndarray,
    photon_array: np.ndarray,
    sigma_array: np.ndarray,
    image_height: int,
    image_width: int,
    pixel_size: float,
    bleaching: np.ndarray
):
    """
    Generate an image from localized emitters using multi-threading.

    Parameters
    ----------
    xc_array : array_like
        Array of x-coordinates of the emitters.
    yc_array : array_like
        Array of y-coordinates of the emitters.
    photon_array : array_like
        Array of photon counts for each emitter.
    sigma_array : array_like
        Array of standard deviations (sigmas) for each emitter.
    image_height : int
        Height of the output image in pixels.
    image_width : int
        Width of the output image in pixels.
    pixel_size : float
        Size of each pixel in the image.

    Returns
    -------
    Image : ndarray
        2D array representing the generated image.

    Notes
    -----
    The function utilizes multi-threading for parallel processing using Numba's
    `njit` decorator with `parallel=True`. Emitters with non-positive photon
    counts or non-positive sigma values are ignored. Only emitters within a
    distance of 4 sigma from the center of the pixel are considered to save
    computation time.

    The calculation involves error functions (`erf`) to determine the contribution
    of each emitter to the pixel intensity.

    Originally from: https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/Deep-STORM_2D_ZeroCostDL4Mic.ipynb
    """
    Image = np.zeros((image_height, image_width))
    for ij in prange(image_height * image_width):
        j = int(ij / image_width)
        i = ij - j * image_width
        for xc, yc, photon, sigma in zip(xc_array, yc_array, photon_array, sigma_array):
            # Don't bother if the emitter has photons <= 0 or if Sigma <= 0
            if (photon > 0) and (sigma > 0):
                S = sigma * math.sqrt(2)
                x = i * pixel_size - xc
                y = j * pixel_size - yc
                # Don't bother if the emitter is further than 4 sigma from the centre of the pixel
                if (x + pixel_size / 2) ** 2 + (
                    y + pixel_size / 2
                ) ** 2 < 16 * sigma**2:
                    ErfX = math.erf((x + pixel_size) / S) - math.erf(x / S)
                    ErfY = math.erf((y + pixel_size) / S) - math.erf(y / S)
                    Image[j][i] += 0.25 * photon * ErfX * ErfY * bleaching[j][i]
    return Image


def binary2locs(img: np.ndarray, density: float):
    """
    Selects a subset of locations from a binary image based on a specified density.

    Parameters
    ----------
    img : np.ndarray
        2D binary image array where 1s indicate points of interest.
    density : float
        Proportion of points to randomly select from the points of interest.
        Should be a value between 0 and 1.

    Returns
    -------
    filtered_locs : tuple of np.ndarray
        Tuple containing two arrays. The first array contains the row indices
        and the second array contains the column indices of the selected points.

    Notes
    -----
    The function identifies all locations in the binary image where the value is 1.
    It then randomly selects a subset of these locations based on the specified
    density and returns their coordinates.
    """
    all_locs = np.nonzero(img == 1)
    n_points = int(len(all_locs[0]) * density)
    selected_idx = np.random.choice(len(all_locs[0]), n_points, replace=False)
    filtered_locs = all_locs[0][selected_idx], all_locs[1][selected_idx]
    return filtered_locs
