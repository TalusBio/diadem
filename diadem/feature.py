"""Functions for feature finding"""
from typing import List, Union, Tuple
from dataclasses import dataclass

import numpy as np
import numba as nb
from scipy import signal

from . import utils

FWHM_RATIO = 2.35482004503


@dataclass
class Feature:
    """A chromatographic feature for a specific m/z"""

    query_mz: int
    query_rt: float
    mean_mz: int
    ret_times: np.ndarray
    intensities: np.ndarray
    lower_bound: int = 0
    upper_bound: int = -1

    def __post_init__(self):
        """Initialize calculated values"""
        self._background = None
        self._area = None
        self._peak = None

    @property
    def ret_time_bounds(self):
        """The min and max retention times."""
        return self.ret_times.min(), self.ret_times.max()

    @property
    def lower_bound(self):
        """The index of the lower bound for integration"""
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value):
        """Set the lower bound."""
        self._reset()
        self._lower_bound = value

    @property
    def upper_bound(self):
        """The index of the upper bound for integration."""
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value):
        """Set the upper bound"""
        self._reset()
        self._upper_bound = value

    @property
    def area(self):
        """The integrated peak area"""
        if (self._area is None) or (self._background is None):
            self._area = integrate(
                self.peak, self.ret_times, self.lower_bound, self.upper_bound
            )

        return self._area

    @property
    def background(self):
        """The median background intensity"""
        if self._background is None:
            self._background = calc_background(
                self.intensities,
                self.lower_bound,
                self.upper_bound,
            )

        return self._background

    @property
    def peak(self):
        """The corrected intensities for the integrated region."""
        if self._peak is None:
            self._peak = calc_peak(
                self.intensities,
                self.lower_bound,
                self.upper_bound,
                self.background,
            )

        return self._peak

    def update_bounds(self) -> None:
        """Update the integration boundaries for a feature."""
        center = np.argmax(self.intensities)
        width, *_ = signal.peak_widths(self.intensities, [center], 0.5)
        sigma2 = width * FWHM_RATIO
        self.lower_bound = int(np.floor(center - sigma2))
        self.upper_bound = int(np.ceil(center + sigma2))
        self._reset()

    def _reset(self):
        """Reset the state after updating integration boundaries"""
        self._background = None
        self._area = None
        self._peak = None


@nb.njit
def calc_background(int_array, lower_bound, upper_bound):
    """Calculate the median for the background."""
    mask = np.array([True] * len(int_array))
    mask[lower_bound:upper_bound] = False
    return np.median(int_array[mask])


@nb.njit
def calc_peak(int_array, lower_bound, upper_bound, background=None):
    """Get the background corrected peak"""
    if background is None:
        background = calc_background(int_array, lower_bound, upper_bound)

    peak = int_array - background
    peak[peak < 0] = 0
    return peak[lower_bound:upper_bound]


@nb.njit
def integrate(peak, ret_times, lower_bound, upper_bound):
    """Perform trapzoidal integration.

    I tested this to be ~10x faster than numpy alone.
    """
    peak_rt = ret_times[lower_bound:upper_bound]
    return np.trapz(peak, peak_rt)


@nb.njit
def extract_feature(
    query_mz: int,
    query_rt: float,
    mz_array: np.ndarray,
    int_array: np.ndarray,
    rt_array: np.ndarray,
    mz_tol: float,
    rt_tol: float,
) -> List[int]:
    """Quickly get the indices of feature peaks.

    Returns
    -------
    numpy.ndarray
        The indices of the matching peak.
    """
    mz_tol = int(mz_tol * query_mz / 1e6)
    mz_bounds = (query_mz - mz_tol, query_mz + mz_tol)
    rt_bounds = (query_rt - rt_tol, query_rt + rt_tol)

    # Because the data should be sorted by rt then intensity,
    # the first peak encountered at each m/z is the most intense.
    feat_xic = []
    feat_mz = []
    feat_rt = []
    feat_idx = []
    used_rt = False
    arrays = zip(mz_array, rt_array, int_array)
    for idx, (mz, rt, int_) in enumerate(arrays):
        if rt > rt_bounds[1]:
            break

        if rt < rt_bounds[0]:
            continue

        if not feat_rt or rt != feat_rt[-1]:
            if len(feat_rt) != len(feat_xic):
                feat_xic.append(0)

            feat_rt.append(rt)
            used_rt = False

        if used_rt or mz < mz_bounds[0] or mz > mz_bounds[1]:
            continue

        feat_mz.append(mz)
        feat_xic.append(int_)
        feat_idx.append(idx)
        used_rt = True

    # If the m/z wasnt found in the last scan:
    if len(feat_rt) != len(feat_xic):
        feat_xic.append(0)

    if not feat_mz:
        return None, None

    feature = (
        np.array(feat_mz).mean(),
        np.array(feat_rt),
        np.array(feat_xic),
    )

    return feature, feat_idx
