"""Score functions"""
import numpy as np
import numba as nb


@nb.njit
def mini_hyperscore(intensities):
    """Like the X!Tandem Hyperscore...

    ... but with an exponential instead of factorial

    Parameters
    ----------
    intensities : np.ndarray
        The intensities. Zero indicates missing.

    Returns
    -------
    float
        The score.
    """
    n_peaks = (intensities > 0).sum()
    return np.log(2**n_peaks * intensities.sum() / 2)


@nb.njit
def hyperscore(intensities):
    """A version of the X!Tandem Hyperscore

    Parameters
    ----------
    intensities : np.ndarray
        The intensities. Zero indicates missing.

    Returns
    -------
    float
        The score.
    """
    n_peaks = (intensities > 0).sum()
    return np.log(np.factorial(n_peaks / 2) ** 2 * intensities.sum() / 2)
