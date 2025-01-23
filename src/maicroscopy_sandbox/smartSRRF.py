import numpy as np
from nanopyx import eSRRF
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from nanopyx.core.transform._interpolation import cr_interpolate
from nanopyx.core.transform._le_interpolation_catmull_rom import (
    ShiftAndMagnify,
)
from nanopyx import calculate_decorr_analysis


def smartSRRF(microscope, magnification=4, plot=True):

    interpolator = ShiftAndMagnify()
    running = True

    widefield_image = microscope.acquire_image()
    magnified_image = interpolator.run(
        widefield_image, 0, 0, magnification, magnification
    )
    print(magnified_image.shape)

    rgc_maps = []

    rgc_frame = eSRRF(widefield_image, magnification=magnification)[0]
    rgc_maps = rgc_frame.copy()

    quality = 0

    while running:

        current_rgc = eSRRF(
            microscope.acquire_image(), magnification=magnification
        )[0]
        rgc_maps = np.append(
            rgc_maps,
            current_rgc,
            axis=0,
        )
        current_esrrf = np.mean(rgc_maps, axis=0)
        if plot:
            plt.imshow(
                current_esrrf,
                cmap="gray",
            )
            plt.show()

        current_quality = calculate_decorr_analysis(current_esrrf)
        print(current_quality)

        if current_quality - quality < 0.01:
            running = False
            return current_esrrf
        else:
            quality = current_quality
            continue
