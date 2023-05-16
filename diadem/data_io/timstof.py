from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from ms2ml.utils.mz_utils import get_tolerance
from msflattener.bruker import get_timstof_data
from numpy.typing import NDArray

from diadem.config import DiademConfig
from diadem.data_io.mzml import (
    MassError,
    ScanGroup,
    SpectrumStacker,
    StackedChromatograms,
)
from diadem.data_io.utils import slice_from_center, xic
from diadem.search.metrics import get_ref_trace_corrs
from diadem.utilities.utils import is_sorted

if "PLOTDIADEM" in os.environ:
    pass

IMSError = Literal["abs", "pct"]


@dataclass
class TimsStackedChromatograms(StackedChromatograms):
    """A class containing the elements of a stacked chromatogram.

    The stacked chromatogram is the extracted ion chromatogram
    across a window of spectra.

    Parameters
    ----------
    array :
        An array of shape [i, w]
    mzs :
        An array of shape [i]
    ref_index :
        An integer in the range [0, i]
    ref_ims :
        IMS value for the reference peak
    parent_index :
        Identifier of the range where the window was extracted
    base_peak_intensity :
        Intensity of the base peak in the reference spectrum
    stack_peak_indices :
        List of indices used to stack the array, it is a list of dimensions [w],
        where each element can be either a list of indices or an empty list.

    Details
    -------
    The dimensions of the arrays are `w` the window
    size of the extracted ion chromatogram. `i` the number
    of m/z peaks that were extracted.

    """

    ref_ims: float

    @staticmethod
    # @profile
    def from_group(
        group: TimsScanGroup,
        index: int,
        window: int = 21,
        mz_tolerance: float = 0.02,
        mz_tolerance_unit: MassError = "da",
        ims_tolerance: float = 0.03,
        ims_tolerance_unit: IMSError = "abs",
        # TODO implement abs and pct ims error ...
        # maybe just use the mz ppm tolerance an multiply by 10000 ...
        min_intensity_ratio: float = 0.01,
        min_correlation: float = 0.5,
        max_peaks: int = 150,
    ) -> TimsStackedChromatograms:
        """Generates a stacked chromatogram from a TimsScanGroup.

        Parameters
        ----------
        group : ScanGroup
            A scan group containing the spectra to stack
        index : int
            The index of the spectrum to use as the reference
        window : int, optional
            The number of spectra to stack, by default 21
        mz_tolerance : float, optional
            The tolerance to use when matching m/z values, by default 0.02
        mz_tolerance_unit : MassError, optional
            The unit of the tolerance, by default "da"
        ims_tolerance : float, optional
            The tolerance to use for the ion mobility dimension.
        ims_tolerance_unit : IMSError
            The unit of the IMS tolerance to use, 'pct' (percent) and
            'abs' (absolute) are acceptable values.
        min_intensity_ratio : float, optional
            The minimum intensity ratio to use when stacking, by default 0.01
        min_correlation : float, optional
            The minimum correlation to use when stacking, by default 0.5
        max_peaks : int, optional
            The maximum number of peaks to return in a group, by default is 150.
            If the candidates is more than this number, it will the best co-eluting
            peaks.

        """
        # TODO consider if deisotoping should happen at this stage,
        # not at the pre-processing stage.
        # The issue with that was that tracking indices is harder.
        # TODO abstract this so it is not as redundant with super().from_group()

        # The center index is the same as the provided index
        # Except in cases where the edge of the group is reached, where
        # the center index is adjusted to the edge of the group
        slice_q, center_index = slice_from_center(
            center=index,
            window=window,
            length=len(group.mzs),
        )
        mzs = group.mzs[slice_q]
        intensities = group.intensities[slice_q]
        imss = group.imss[slice_q]

        center_mzs = mzs[center_index]
        center_intensities = intensities[center_index]
        center_ims = imss[center_index]

        bp_intensity_index = np.argmax(center_intensities)
        bp_ims = center_ims[bp_intensity_index]

        if ims_tolerance_unit != "abs":
            raise NotImplementedError()
        ims_keep = np.abs(center_ims - bp_ims) <= ims_tolerance

        center_mzs = center_mzs[ims_keep]
        center_intensities = center_intensities[ims_keep]
        assert is_sorted(center_mzs)

        # TODO move this to its own helper function (collapse_unique ??)
        # ... Maybe even "proprocess_ims_spec(mzs, imss, ints, ref_ims, ...)"
        # We collapse all unique mzs, after filtering for IMS tolerance
        # Note: Getting what indices were used to generate u_mzs[0]
        # would be np.where(inv == 0)
        # TODO testing equality and uniqueness on floats might not be wise.
        # I should change this to an int ... maybe setting a "intensity bin"
        # value like comet does (0.02??)
        u_center_mzs, u_center_intensities, inv = _bin_spectrum_intensities(
            center_mzs,
            center_intensities,
            bin_width=0.001,
            bin_offset=0,
        )
        assert is_sorted(u_center_mzs)

        xic_outs = []

        for i, (m, inten, ims) in enumerate(zip(mzs, intensities, imss)):
            # We first filter the peaks that are inside our IMS tolerance
            # By getting their indices.
            t_int_keep = np.abs(ims - bp_ims) <= ims_tolerance
            t_int_keep = np.where(t_int_keep)[0]

            m = m[t_int_keep]
            inten = inten[t_int_keep]

            u_mzs, u_intensities, inv = _bin_spectrum_intensities(
                m,
                inten,
                bin_width=0.001,
                bin_offset=0,
            )

            outs, inds = xic(
                query_mz=u_mzs,
                query_int=u_intensities,
                mzs=u_center_mzs,
                tolerance=mz_tolerance,
                tolerance_unit=mz_tolerance_unit,
            )

            # Since inds are the indices used from that suvset array;
            # We find what indices in the original array were used for
            # each value.
            out_inds = []
            for y in inds:
                if len(y) > 0:
                    collapsed_indices = np.concatenate(
                        [np.where(inv == w)[0] for w in y],
                    )
                    out_inds.append(np.unique(t_int_keep[collapsed_indices]))
                else:
                    out_inds.append([])

            xic_outs.append((outs, out_inds))
            if i == center_index:
                assert xic_outs[-1][0].sum() >= u_center_intensities.max()

        stacked_arr = np.stack([x[0] for x in xic_outs], axis=-1)

        indices = [x[1] for x in xic_outs]

        if stacked_arr.shape[-2] > 1:
            ref_id = np.argmax(stacked_arr[..., center_index])
            corrs = get_ref_trace_corrs(arr=stacked_arr, ref_idx=ref_id)

            # I think adding the 1e-5 is needed here due to numric instability
            # in the flaoting point operation
            assert np.max(corrs) <= (
                corrs[ref_id] + 1e-5
            ), "Reference does not have max corrr"

            max_peak_corr = np.sort(corrs)[-max_peaks] if len(corrs) > max_peaks else -1
            keep = corrs >= max(min_correlation, max_peak_corr)
            keep_corrs = corrs[keep]

            stacked_arr = stacked_arr[..., keep, ::1]
            u_center_mzs = u_center_mzs[keep]
            u_center_intensities = u_center_intensities[keep]
            indices = [[y for y, k in zip(x, keep) if k] for x in indices]
        else:
            keep_corrs = np.array([1.0])

        ref_id = np.argmax(stacked_arr[..., center_index])
        bp_int = stacked_arr[ref_id, center_index]
        # TODO: This might be a good place to plot the stacked chromatogram

        out = TimsStackedChromatograms(
            array=stacked_arr,
            mzs=u_center_mzs,
            ref_index=ref_id,
            parent_index=index,
            base_peak_intensity=bp_int,
            stack_peak_indices=indices,
            center_intensities=u_center_intensities,
            ref_ims=bp_ims,
            correlations=keep_corrs,
        )
        return out


def _bin_spectrum_intensities(
    mzs: NDArray,
    intensities: NDArray,
    bin_width: float = 0.02,
    bin_offset: float = 0.0,
) -> tuple[NDArray, NDArray, list[list[int]]]:
    """Bins the intensities based on the mz values.

    Returns
    -------
    new_mzs:
        The new mz array
    new_intensities
        The new intensity array
    inv:
        Index for the new mzs in the original mzs and intensities
        Note: Getting what indices were used to generate new_mzs[0]
        would be np.where(inv == 0)

    """
    new_mzs, inv = np.unique(
        np.rint((mzs + bin_offset) / bin_width),
        return_inverse=True,
    )
    new_mzs = (new_mzs * bin_width) - bin_offset
    new_intensities = np.zeros(len(new_mzs), dtype=intensities.dtype)
    np.add.at(new_intensities, inv, intensities)
    return new_mzs, new_intensities, inv


@dataclass
class TimsScanGroup(ScanGroup):
    """Represent all 'spectra' that share an isolation window."""

    imss: list[NDArray]
    precursor_imss: list[NDArray]

    def __post_init__(self) -> None:
        """Validates that the values in the instance are consistent.

        Automatically runs when a new instance is created.
        """
        super().__post_init__()
        if len(self.imss) != len(self.mzs):
            raise ValueError("IMS values do not have the same lenth as the MZ values")

    @classmethod
    def _elems_from_fragment_cache(cls, file):
        elems, data = super()._elems_from_fragment_cache(file)
        elems["imss"] = data["imss"]
        return elems, data

    @classmethod
    def _precursor_elems_from_cache(cls, file):
        elems, data = super()._precursor_elems_from_cache(file)
        elems["imss"] = data["imss"]
        return elems, data

    def to_cache(self, Path):
        """Saves the group to a cache file."""
        super().to_cache(Path)

    def as_dataframe(self) -> pl.DataFrame:
        """Returns a dataframe with the data in the group.

        The dataframe has the following columns:
        - mzs: list of mzs for each spectrum
        - intensities: list of intensities for each spectrum
        - retention_times: retention times for each spectrum
        - precursor_start: start of the precursor range
        - precursor_end: end of the precursor range
        - ims: list of ims values for each spectrum
        """
        out = super().as_dataframe()
        out["ims"] = self.imss
        return out

    def precursor_dataframe(self) -> pl.DataFrame:
        df = super().precursor_dataframe()
        df = df.with_columns(
            pl.Series(name="precursor_imss", values=self.precursor_imss),
        )
        return df

    def get_highest_window(
        self,
        window: int,
        min_intensity_ratio: float,
        mz_tolerance: float,
        mz_tolerance_unit: MassError,
        ims_tolerance: float,
        ims_tolerance_unit: IMSError,
        min_correlation: float,
        max_peaks: int,
    ) -> TimsStackedChromatograms:
        """Gets the highest intensity window of the chromatogram.

        Briefly ...
        1. Gets the highes peak accross all spectra in the chromatogram range.
        2. Finds what peaks are in that same spectrum.
        3. Looks for spectra around that spectrum.
        4. extracts the chromatogram for all mzs in the "parent spectrum"

        """
        top_index = np.argmax(self.base_peak_int)
        window = TimsStackedChromatograms.from_group(
            self,
            window=window,
            index=top_index,
            min_intensity_ratio=min_intensity_ratio,
            min_correlation=min_correlation,
            mz_tolerance=mz_tolerance,
            mz_tolerance_unit=mz_tolerance_unit,
            ims_tolerance=ims_tolerance,
            ims_tolerance_unit=ims_tolerance_unit,
            max_peaks=max_peaks,
        )

        return window

    # This is an implementation of a method used by the parent class
    def _scale_spectrum_at(
        self,
        spectrum_index: int,
        value_indices: NDArray[np.int64],
        scaling_factor: float,
    ) -> None:
        i = spectrum_index  # Alias for brevity in within this function

        if len(value_indices) > 0:
            self.intensities[i][value_indices] = (
                self.intensities[i][value_indices] * scaling_factor
            )
        else:
            return None

        # TODO this is hard-coded right now, change as a param
        int_remove = self.intensities[i] < 10
        if np.any(int_remove):
            self.intensities[i] = self.intensities[i][np.invert(int_remove)]
            self.mzs[i] = self.mzs[i][np.invert(int_remove)]
            self.imss[i] = self.imss[i][np.invert(int_remove)]

        if len(self.intensities[i]):
            self.base_peak_int[i] = np.max(self.intensities[i])
            self.base_peak_mz[i] = self.mzs[i][np.argmax(self.intensities[i])]
        else:
            self.base_peak_int[i] = 0
            self.base_peak_mz[i] = 0

    def __len__(self) -> int:
        """Returns the number of spectra in the group."""
        return len(self.imss)


class TimsSpectrumStacker(SpectrumStacker):
    """Helper class that stacks the spectra of TimsTof file into chromatograms."""

    def __init__(self, filepath: PathLike, config: DiademConfig) -> None:
        """Initializes the class.

        Parameters
        ----------
        filepath : PathLike
            Path to the TimsTof file
        config : DiademConfig
            Configuration object
        """
        self.filepath = filepath
        self.config = config
        self.cache_location = Path(filepath).with_suffix(".msms.parquet")
        # self.cache_location = Path(filepath).with_suffix(".centroided.parquet")
        if self.cache_location.exists():
            logger.info(f"Found cache file at {self.cache_location}")
        else:
            df = get_timstof_data(filepath, centroid=False)
            df.write_parquet(self.cache_location)
            del df

        unique_windows = (
            pl.scan_parquet(self.cache_location)
            .select(pl.col(["quad_low_mz_values", "quad_high_mz_values"]))
            .filter(pl.col("quad_low_mz_values") > 0)
            .sort("quad_low_mz_values")
            .unique()
            .collect()
        )

        if "DEBUG_DIADEM" in os.environ:
            logger.error("RUNNING DIADEM IN DEBUG MODE (only the 4th precursor index)")
            self.unique_precursor_windows = unique_windows[3:4].rows(named=True)
        else:
            self.unique_precursor_windows = unique_windows.rows(named=True)

    @contextmanager
    def lazy_datafile(self) -> pl.LazyFrame:
        """Scans the cached version of the data and yields it as a context manager."""
        yield pl.scan_parquet(self.cache_location)

    # @profile
    def _precursor_iso_window_groups(
        self,
        precursor_window: dict[str, float],
    ) -> dict[str:TimsScanGroup]:
        elems = self._precursor_iso_window_elements(precursor_window)
        prec_info = self._precursor_iso_window_elements(
            {"quad_low_mz_values": -1, "quad_high_mz_values": -1},
            mz_range=list(precursor_window.values()),
        )

        assert is_sorted(prec_info["retention_times"])

        out = TimsScanGroup(
            precursor_mzs=prec_info["mzs"],
            precursor_intensities=prec_info["intensities"],
            precursor_retention_times=prec_info["retention_times"],
            precursor_imss=prec_info["imss"],
            **elems,
        )
        return out

    # @profile
    def _precursor_iso_window_elements(
        self,
        precursor_window: dict[str, float],
        mz_range: None | tuple[float, float] = None,
    ) -> dict[str : dict[str:NDArray]]:
        with self.lazy_datafile() as datafile:
            datafile: pl.LazyFrame
            promise = (
                pl.col("quad_low_mz_values") == precursor_window["quad_low_mz_values"]
            ) & (
                pl.col("quad_high_mz_values") == precursor_window["quad_high_mz_values"]
            )
            ms_data = datafile.filter(promise).sort("rt_values")

            if mz_range is not None:
                nested_cols = [
                    "mz_values",
                    "corrected_intensity_values",
                    "mobility_values",
                ]
                non_nested_cols = [
                    x for x in ms_data.head().collect().columns if x not in nested_cols
                ]
                ms_data = (
                    ms_data.explode(nested_cols)
                    .filter(pl.col("mz_values").is_between(mz_range[0], mz_range[1]))
                    .groupby(pl.col(non_nested_cols))
                    .agg(nested_cols)
                    .sort("rt_values")
                )

            ms_data = ms_data.collect()

            bp_indices = [np.argmax(x) for x in ms_data["corrected_intensity_values"]]
            bp_ints = [
                x1.to_numpy()[x2]
                for x1, x2 in zip(ms_data["corrected_intensity_values"], bp_indices)
            ]
            bp_ints = np.array(bp_ints)
            bp_mz = [
                x1.to_numpy()[x2] for x1, x2 in zip(ms_data["mz_values"], bp_indices)
            ]
            bp_mz = np.array(bp_mz)
            bp_indices = np.array(bp_indices)
            rts = ms_data["rt_values"].to_numpy(zero_copy_only=True)
            assert is_sorted(rts)

            quad_high = ms_data["quad_high_mz_values"][0]
            quad_low = ms_data["quad_low_mz_values"][0]
            window_name = str(quad_low) + "_" + str(quad_high)

            template = window_name + "_{}"
            scan_indices = [template.format(i) for i in range(len(rts))]

            x = {
                "precursor_range": (quad_low, quad_high),
                "base_peak_int": bp_ints,
                "base_peak_mz": bp_mz,
                # 'base_peak_ims':bp_ims,
                "iso_window_name": window_name,
                "retention_times": rts,
                "scan_ids": scan_indices,
            }
            orders = [np.argsort(x.to_numpy()) for x in ms_data["mz_values"]]

            x.update(
                {
                    "mzs": [
                        x.to_numpy()[o] for x, o in zip(ms_data["mz_values"], orders)
                    ],
                    "intensities": [
                        x.to_numpy()[o]
                        for x, o in zip(ms_data["corrected_intensity_values"], orders)
                    ],
                    "imss": [
                        x.to_numpy()[o]
                        for x, o in zip(ms_data["mobility_values"], orders)
                    ],
                },
            )

        return x

    def get_iso_window_groups(self, workerpool: None | Parallel) -> list[TimsScanGroup]:
        """Get scan groups for each unique isolation window.

        Parameters
        ----------
        workerpool : None | Parallel
            If None, the function will be run in serial mode.
            If Parallel, the function will be run in parallel mode.
            The Parallel is created using joblib.Parallel.

        Returns
        -------
        list[TimsScanGroup]
            A list of TimsScanGroup objects.
            Each of them corresponding to an unique isolation window from
            the quadrupole.
        """
        if workerpool is None:
            results = [
                self._precursor_iso_window_groups(i)
                for i in self.unique_precursor_windows
            ]
        else:
            results = workerpool(
                delayed(self._precursor_iso_window_groups)(i)
                for i in self.unique_precursor_windows
            )

        return results

    def yield_iso_window_groups(self) -> Iterator[TimsScanGroup]:
        """Yield scan groups for each unique isolation window."""
        for i in self.unique_precursor_windows:
            results = self._precursor_iso_window_groups(i)
            yield results


# @profile
def find_neighbors_mzsort(
    ims_vals: NDArray[np.float32],
    sorted_mz_values: NDArray[np.float32],
    intensities: NDArray[np.float32],
    top_n: int = 500,
    top_n_pct: float = 0.1,
    ims_tol: float = 0.02,
    mz_tol: float = 0.02,
    mz_tol_unit: MassError = "Da",
) -> dict[int : list[int]]:
    """Finds the neighbors of the most intense peaks.

    It finds the neighboring peaks for the `top_n` most intense peaks
    or the (TOTAL_PEAKS) * `top_n_pct` peaks, whichever is largest.

    Arguments:
    ---------
    ims_vals: NDArray[np.float32]
        Array containing the ion mobility values of the precursor.
    sorted_mz_values: NDArray[np.float32]
        Sorted array contianing the mz values
    intensities: NDArray[np.float32]
        Array containing the intensities of the peaks
    top_n : int
        Number of peaks to use as seeds for neighborhood finding,
        defautls to 500.
        It will internally use the largest of either this number
        or `len(intensities)*top_n_pct`
    top_n_pct : float
        Minimum percentage of the intensities to use as seeds for the
        neighbors. defaults to 0.1
    ims_tol : float
        Maximum distance to consider as a neighbor along the IMS dimension.
        defaults to 0.02
    mz_tol : float
        Maximum distance to consider as a neighbor along the MZ dimension.
        defaults to 0.02
    mz_tol_unit : Literal['ppm', 'Da']
        Unit that describes the mz tolerance.
        defaults to 'Da'

    """
    if mz_tol_unit.lower() == "da":
        pass
    elif mz_tol_unit.lower() == "ppm":
        mz_tol: NDArray[np.float32] = get_tolerance(
            mz_tol,
            theoretical=sorted_mz_values,
            unit="ppm",
        )
    else:
        raise ValueError("Only 'Da' and 'ppm' values are supported as mass errors")

    top_n = int(max(len(intensities) * top_n_pct, top_n))
    if len(intensities) > top_n:
        top_indices = np.argpartition(intensities, -top_n)[-top_n:]
    else:
        top_indices = None

    opts = {}
    for i1, (ims1, mz1) in enumerate(zip(ims_vals, sorted_mz_values)):
        if top_indices is not None and i1 not in top_indices:
            opts.setdefault(i1, []).append(i1)
            continue

        candidates = np.arange(
            np.searchsorted(sorted_mz_values, mz1 - mz_tol),
            np.searchsorted(sorted_mz_values, mz1 + mz_tol, side="right"),
        )

        tmp_ims = ims_vals[candidates]

        match_indices = np.abs(tmp_ims - ims1) <= ims_tol
        match_indices = np.where(match_indices)[0]
        for i2 in candidates[match_indices]:
            opts.setdefault(i1, [i1]).append(i2)
            opts.setdefault(i2, [i2]).append(i1)

    opts = {k: list(set(v)) for k, v in opts.items()}
    return opts


def get_break_indices(
    inds: NDArray[np.int64],
    min_diff: float | int = 1,
    break_values: NDArray = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Gets the incides and break values for an increasing array.

    Example:
    -------
    >>> tmp = np.array([1,2,3,4,7,8,9,11,12,13])
    >>> bi = get_break_indices(tmp)
    >>> bi
    (array([ 0, 4, 7, 10]), array([ 1,  7, 11, 13]))
    >>> [tmp[si: ei] for si, ei in zip(bi[0][:-1], bi[0][1:])]
    [array([1, 2, 3, 4]), array([7, 8, 9]), array([11, 12, 13])]
    """
    if break_values is None:
        break_values = inds
    breaks = 1 + np.where(np.diff(break_values) > min_diff)[0]
    breaks = np.concatenate([np.array([0]), breaks, np.array([inds.size - 1])])

    break_indices = inds[breaks]
    breaks[-1] += 1

    return breaks, break_indices
