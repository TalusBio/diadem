"""Functions for feature finding"""
import numpy as np
import numba as nb
from scipy import signal

FWHM_RATIO = 2 / 2.35482004503


def build(intensities):
    """Find the boundaries of a feature from the intensities.

    This function finds the FWHM and uses it to estimate the $\sigma$
    for 95% of a Gaussian.

    Parameters
    ----------
    intensities : array of float
        The intensities of the chromatographic peak.

    Returns
    -------
    start : int
        The starting index of the peak.
    finish : int
        The ending index of the peak.
    """
    center = np.argmax(intensities)
    width, *_ = signal.peak_widths(intensities, [center], 0.5)
    sigma = width * FWHM_RATIO
    return int(np.floor(center - sigma)), int(np.ceil(center + sigma))


def integrate(intensities, ret_times, boundaries=None):
    """Calculate the peak area of a feature.

    Parameters
    ----------
    intensities : array of flaot
        The intensity vector.
    ret_times : array of float
        The retention times.
    boundaries : tuple of int, optional
        The indices of the integration boundaries.
    """
    if boundaries is None:
        boundaries = build(intensities)

    background = np.concatenate(
        [
            intensities[: boundaries[0]],
            intensities[boundaries[1] :],
        ]
    )

    intensities -= np.median(background)
    peak_int = intensities[boundaries[0] : boundaries[1]]
    peak_rt = intensities[boundaries[0] : boundaries[1]]
    return np.trapz(peak_int, peak_rt)
