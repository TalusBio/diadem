"""A SQLite3 databse for DIA mass spectrometry data"""
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass

import numba as nb
import numpy as np
import awkward as ak
from tqdm.auto import tqdm
from pyteomics.mzml import MzML

from . import utils
from .feature import Feature

LOGGER = logging.getLogger(__name__)


class DiaRun:
    """A the mass spectra from a DIA mass spectrometry run.

    Parameters
    ----------
    ms_data_file : str or Path
        The mass spectrometry data file to parse. Currently, only mzML files
        are supported.
    """

    def __init__(self, ms_data_file):
        self._ms_data_file = Path(ms_data_file)
        self.precursors = None
        self.windows = None

    def parse(self):
        """Parse the mzML file.

        Returns
        -------
        self
        """
        self.precursors = DiaWindow()
        self.windows = {}

        LOGGER.info("Reading '%s'...", self._ms_data_file)
        with MzML(str(self._ms_data_file)) as mzdata:
            for spectrum in tqdm(mzdata, unit="spectra"):
                ret_time = spectrum["scanList"]["scan"][0]["scan start time"]
                mz_array = spectrum["m/z array"]
                int_array = spectrum["intensity array"]
                if spectrum["ms level"] == 1:
                    self.precursors.add_spectrum(mz_array, int_array, ret_time)
                    continue

                window = spectrum["precursorList"]["precursor"][0][
                    "isolationWindow"
                ]
                center = window["isolation window target m/z"]
                isolation_window = (
                    center - window["isolation window lower offset"],
                    center + window["isolation window upper offset"],
                )

                try:
                    parsed = self.windows[isolation_window]
                except KeyError:
                    parsed = DiaWindow(isolation_window)
                    self.windows[isolation_window] = parsed

                parsed.add_spectrum(mz_array, int_array, ret_time)

        return self


class DiaWindow:
    """Data from a single DIA isolation window.

    Parameters
    ----------
    isolation_window : tuple of (float, float) or None, optional
        The precurosr isolation window. ``None`` indicates precursor (MS1)
        scans.
    """

    def __init__(self, isolation_window=None, precision=5):
        """Initialize the DiaWindow"""
        self.isolation_window = isolation_window
        self.precision = 5
        self.mz_arrays = ak.ArrayBuilder()
        self.intensity_arrays = ak.ArrayBuilder()
        self.ret_times = ak.ArrayBuilder()

        # Only create after adding all of the spectra:
        self.order = None
        self.offsets = None

        # Only used when searching:
        self.consumed = None
        self.mask = None
        self.position = None

    def __getitem__(self, idx):
        """Retrieve spectra"""
        return (
            self.mz_arrays[idx],
            self.intensity_arrays[idx],
            self.ret_times[idx],
        )

    def __enter__(self):
        """Enter for searching."""
        # Sort peaks by intensity.
        self.order = np.argsort(-ak.flatten(self.intensity_arrays.snapshot()))
        self.offsets = np.array(self.mz_arrays.snapshot().layout.starts)
        self.consumed = np.zeros(len(self.order), dtype=bool)
        self.mask = ak.unflatten(
            ak.Array(self.consumed),
            ak.num(self.mz_arrays.snapshot()),
        )
        self.position = 0
        return self

    def __exit__(self, *args):
        """Reset after searching."""
        self.consumed = None
        self.order = None
        self.mask = None
        self.offsets = None
        self.position = 0

    def __iter__(self):
        """Iterate through remaining peaks."""
        while True:
            try:
                yield next(self)
            except StopIteration:
                break

    def __next__(self):
        """Get the next viable peak."""
        try:
            peak = self.order[self.position]
            self.position += 1
            if self.consumed[peak]:
                return next(self)

            return peak
        except ValueError:
            raise StopIteration

    def __len__(self):
        """Get the number of peaks"""
        return len(self.order)

    def peak(self, idx):
        """Return information about a peak.

        Parameters
        ----------
        idx : int
            The 1D index of a peak.

        Returns
        -------
        tuple of float
            The m/z, intensity, and retention time of the peak.
        """
        loc = _idx2loc(idx, self.offsets)
        return (
            self.mz_arrays[loc],
            self.intensity_arrays[loc],
            self.ret_times[loc[0]],
        )

    def consume(self, peaks):
        """Consume a peak.

        Parameters
        ----------
        peaks : int or Slice
            The index of the peak to consume.
        """
        self.consumed[peaks] = True

    def xic(self, moverz, rt_min=None, rt_max=None, tol=10):
        """Create an extracted ion chromatogram.

        Parameters
        ----------
        moverz : int
            The integerized m/z to extract.
        rt_min : float, optional
            The lower retention time bound.
        rt_max : float, optional
            The upper retention time bound.
        tol : float, optional
            The m/z tolerance in parts-per-million (ppm).
        """
        rt_min = 0 if rt_min is None else rt_min
        rt_max = max(self.ret_times) if rt_max is None else rt_max

        offsets = self.offsets
        if offsets is None:
            offsets = np.array(self.mz_arrays.snapshot().layout.starts)

        if self.mask is not None:
            mask = self.mask
        else:
            mask = np.index_exp[:]

        feature_data = _extract_feature(
            moverz=moverz,
            rt_min=rt_min,
            rt_max=rt_max,
            tol=tol,
            mz_arrays=self.mz_arrays.snapshot(),
            int_arrays=self.intensity_arrays.snapshot(),
            ret_times=self.ret_times.snapshot(),
            offsets=offsets,
        )
        return Feature(moverz, *feature_data)

    def add_spectrum(self, mz_array, intensity_array, ret_time):
        """Add a mass spectrum.

        Parameters
        ----------
        mz_array : np.ndarray
            The m/z values.
        intensity_array : np.ndarray
            The intensity values.
        ret_time : float
            The retention time.
        """
        self.mz_arrays.append(mz_array)
        self.intensity_arrays.append(intensity_array)
        self.ret_times.append(float(ret_time))


@nb.njit
def _idx2loc(idx, offsets):
    """Get the location of a peak from the 1D index.

    Parameters
    ----------
    idx : int
        The index of the peak in the flattened array.
    offsets : np.ndarray
        The offsets indicating the start of each spectrum.

    Returns
    -------
    spec_idx : int
        The spectrum that the peak is from.
    peak_idx : int
        The index of the peak within the spectrum.
    """
    spec_idx = np.searchsorted(offsets, idx, side="right") - 1
    peak_idx = idx - offsets[spec_idx]
    return spec_idx, peak_idx


@nb.njit
def _extract_feature(
    moverzs,
    rt_min,
    rt_max,
    tol,
    mz_arrays,
    int_arrays,
    ret_times,
    offsets,
):
    """Extract the ion chromatogram.

    Parameters
    ----------
    moverzs : np.ndarray of float
        The m/z values to extract.
    rt_min : float, optional
        The lower retention time bound.
    rt_max : float, optional
        The upper retention time bound.
    tol : float, optional
        The m/z tolerance in parts-per-million (ppm).
    mz_arrays : awkward.Array
        The m/z arrays
    int_arrays : awkward.Array
        The intensity arrays.
    ret_times : awkward.Array
        The retention time array.

    Returns
    -------
    peak_array : np.ndarray
        The indicies of peaks used.
    rt_array : np.ndarray
        The retention times of the peaks.
    mz_array : np.ndarray
        The m/z values of the peaks.
    int_array : np.ndarray
        The intensities of the peaks.
    """
    mz_tol = tol * moverz / 1e6
    mz_min = moverz - mz_tol
    mz_max = moverz + mz_tol

    ret_times = np.array(ret_times)
    within_rt = np.where((ret_times >= rt_min) & (ret_times <= rt_max))[0]
    rt_array = ret_times[within_rt]

    dims = (moverzs.shape[0], rt_array.shape[0])
    peak_array = np.empty(dims, dtype="int")
    peak_array[:] = None
    mz_array = np.zeros(dims)
    int_array = np.zeros(dims)

    within_rt = set(within_rt)
    for idx, (mz_spec, int_spec) in enumerate(zip(mz_arrays, int_arrays)):
        if idx not in within_rt:
            continue

        mz_spec = np.array(mz_spec)
        int_spec = np.array(int_spec)
        within_mz = np.where((mz_spec >= mz_min) & (mz_spec <= mz_max))[0]
        if not len(within_mz):
            mz_array.append(0)
            int_array.append(0)
            continue

        max_peak = np.argmax(int_spec[within_mz])
        max_idx = within_mz[max_peak]
        mz_array.append(mz_spec[max_idx])
        int_array.append(int_spec[max_idx])
        peak_array.append(offsets[idx] + max_idx)

    return (
        np.array(peak_array),
        rt_array,
        np.array(mz_array),
        np.array(int_array),
    )
