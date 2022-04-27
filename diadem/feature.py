"""Functions for feature finding"""
from typing import List, Union
from dataclasses import dataclass

import numpy as np
from scipy import signal

FWHM_RATIO = 2 / 2.35482004503


@dataclass
class Feature:
    """A chromatographic feature for a specific m/z"""

    query_mz: int
    mean_mz: int
    row_ids: List[int]
    ret_times: np.ndarray
    intensities: np.ndarray
    lower_bound: int = 0
    upper_bound: int = -1

    def __post_init__(self):
        """Initialize calculated values"""
        self._background = None
        self._area = None

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
            self._integrate()

        return self._area

    @property
    def background(self):
        """The median background intensity"""
        if self._background is None:
            mask = np.zeros_like(self.intensities)
            mask[self.lower_bound : self.upper_bound] = 1
            print(mask)
            background = np.ma.array(self.intensities, mask.tolist())
            self._background = np.median(background)

        return self._background

    @property
    def peak(self):
        """The corrected intensities for the integrated region."""
        corrected = self.intensities - self.background
        return corrected[self.lower_bound : self.upper_bound]

    def update_bounds(self) -> None:
        """Update the integration boundaries for a feature."""
        center = np.argmax(self.intensities)
        width, *_ = signal.peak_widths(self.intensities, [center], 0.5)
        sigma = width * FWHM_RATIO
        self.lower_bound = int(np.floor(center - sigma))
        self.upper_bound = int(np.ceil(center + sigma))
        self._reset()

    def _integrate(self) -> None:
        """Calculate the peak area of a feature."""
        peak_rt = self.ret_times[self.lower_bound : self.upper_bound]
        self._area = np.trapz(self.peak[self.peak > 0], peak_rt[self.peak > 0])

    def _reset(self):
        """Reset the state after updating integration boundaries"""
        self._background = None
        self._area = None
