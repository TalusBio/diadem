from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from typing import Literal

import numpy as np
from alphatims.bruker import TimsTOF
from joblib import Parallel, delayed
from loguru import logger
from ms2ml.utils.mz_utils import get_tolerance
from numpy.typing import NDArray
from tqdm.auto import tqdm

from diadem.config import DiademConfig
from diadem.data_io.mzml import (
    MassError,
    ScanGroup,
    SpectrumStacker,
    StackedChromatograms,
)
from diadem.data_io.utils import slice_from_center, xic

# from diadem.deisotoping import deisotope_with_ims
from diadem.search.metrics import get_ref_trace_corrs
from diadem.utilities.utils import is_sorted

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
            bin_width=0.02,
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
                bin_width=0.02,
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

            stacked_arr = stacked_arr[..., keep, ::1]
            u_center_mzs = u_center_mzs[keep]
            u_center_intensities = u_center_intensities[keep]
            indices = [[y for y, k in zip(x, keep) if k] for x in indices]

        ref_id = np.argmax(stacked_arr[..., center_index])
        bp_int = stacked_arr[ref_id, center_index]

        out = TimsStackedChromatograms(
            array=stacked_arr,
            mzs=u_center_mzs,
            ref_index=ref_id,
            parent_index=index,
            base_peak_intensity=bp_int,
            stack_peak_indices=indices,
            center_intensities=u_center_intensities,
            ref_ims=bp_ims,
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
    imss: list[NDArray]
    precursor_imss: list[NDArray]

    def __post_init__(self) -> None:
        """Validates that the values in the instance are consistent.

        Automatically runs when a new instance is created.
        """
        super().__post_init__()
        if len(self.imss) != len(self.mzs):
            raise ValueError("IMS values do not have the same lenth as the MZ values")

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
            self.base_peak_int[i] = -1
            self.base_peak_mz[i] = -1

    def __len__(self) -> int:
        return len(self.imss)


class TimsSpectrumStacker(SpectrumStacker):
    def __init__(self, filepath: PathLike, config: DiademConfig) -> None:
        self.filepath = filepath
        self.config = config
        with self.lazy_datafile() as datafile:
            unique_precursor_indices = np.unique(datafile.precursor_indices)
        unique_precursor_indices = [x for x in unique_precursor_indices if x != 0]

        if "DEBUG_DIADEM" in os.environ:
            logger.error("RUNNING DIADEM IN DEBUG MODE (only the 4th precursor index)")
            self.unique_precursor_indices = unique_precursor_indices[3:4]
        else:
            self.unique_precursor_indices = unique_precursor_indices

    @contextmanager
    def lazy_datafile(self) -> TimsTOF:
        """Reads the timstof data and yields it as context manager.

        This is right now a bandaid to prevent the full TimsTOF object
        to be stored in memmory for the whole lifetime of this class.
        """
        _datafile = TimsTOF(self.filepath)
        yield _datafile
        del _datafile

    def _precursor_iso_window_groups(
        self,
        precursor_index: int,
        progress: bool = True,
    ) -> dict[str:TimsScanGroup]:
        elems = self._precursor_iso_window_elements(precursor_index, progress)
        out = {}
        for k, v in elems.items():
            # Note: precursor id of 0 means inactive quadrupole
            # and collision cell.
            prec_info = self._precursor_iso_window_elements(
                0,
                progress=progress,
                mz_range=v["precursor_range"],
            )
            assert len(list(prec_info)) == 1
            prec_key = list(prec_info)[0]
            prec_info = prec_info[prec_key]
            inten_filter = prec_info["base_peak_int"] > 0

            precursor_mzs = [
                k for i, k in enumerate(prec_info["mzs"]) if inten_filter[i]
            ]
            precursor_intensities = [
                k for i, k in enumerate(prec_info["intensities"]) if inten_filter[i]
            ]
            precursor_imss = [
                k for i, k in enumerate(prec_info["imss"]) if inten_filter[i]
            ]
            precursor_rts = prec_info["retention_times"][inten_filter]

            assert is_sorted(precursor_rts)

            out[k] = TimsScanGroup(
                precursor_mzs=precursor_mzs,
                precursor_intensities=precursor_intensities,
                precursor_rts=precursor_rts,
                precursor_imss=precursor_imss,
                **v,
            )
        return out

    def _precursor_iso_window_elements(
        self,
        precursor_index: int,
        progress: bool = True,
        mz_range: None | tuple[float, float] = None,
    ) -> dict[str : dict[str:NDArray]]:
        with self.lazy_datafile() as datafile:
            index_win_dicts = _get_precursor_index_windows(
                datafile,
                precursor_index=precursor_index,
                progress=progress,
                mz_range=mz_range,
            )
            out = {}

            for k, v in index_win_dicts.items():
                mzs = []
                ints = []
                imss = []

                for vi in v:
                    if len(vi["mz"]) > 0:
                        new_order = np.argsort(vi["mz"])
                        mzs.append(vi["mz"][new_order])
                        ints.append(vi["intensity"][new_order])
                        imss.append(vi["ims"][new_order])
                    else:
                        mzs.append(vi["mz"])
                        ints.append(vi["intensity"])
                        imss.append(vi["ims"])

                # TODO change this to a data structure that stores
                # this only once.
                quad_low = list({x["quad_low_mz_values"] for x in v})
                quad_high = list({x["quad_high_mz_values"] for x in v})
                assert len(quad_high) == 1
                assert len(quad_low) == 1

                bp_indices = [np.argmax(x) if len(x) else None for x in ints]
                bp_ints = [
                    x1[x2] if x2 is not None else 0 for x1, x2 in zip(ints, bp_indices)
                ]
                bp_ints = np.array(bp_ints)
                bp_mz = [
                    x1[x2] if x2 is not None else -1 for x1, x2 in zip(mzs, bp_indices)
                ]
                bp_mz = np.array(bp_mz)
                bp_indices = np.array(bp_indices)
                # bp_ims = np.array([x1[x2] for x1, x2 in zip(imss, bp_indices)])
                rts = np.array([x["rt_values_min"] for x in v])
                scan_indices = [str(x["scan_indices"]) for x in v]

                x = {
                    "precursor_range": (quad_low[0], quad_high[0]),
                    "mzs": mzs,
                    "intensities": ints,
                    "imss": imss,
                    "base_peak_int": bp_ints,
                    "base_peak_mz": bp_mz,
                    # 'base_peak_ims':bp_ims,
                    "iso_window_name": k,
                    "retention_times": rts,
                    "scan_ids": scan_indices,
                }
                out[k] = x

        return out

    def get_iso_window_groups(self, workerpool: None | Parallel) -> list[TimsScanGroup]:
        results = []

        if workerpool is None:
            nested_results = [
                self._precursor_iso_window_groups(i)
                for i in self.unique_precursor_indices
            ]
        else:
            nested_results = workerpool(
                delayed(self._precursor_iso_window_groups)(i)
                for i in self.unique_precursor_indices
            )

        for r in nested_results:
            for rr in r.values():
                results.append(rr)

        return results

    def yield_iso_window_groups(self, progress: bool = True) -> Iterator[TimsScanGroup]:
        for i in self.unique_precursor_indices:
            nested_results = self._precursor_iso_window_groups(i, progress=progress)
            for r in nested_results.values():
                yield r


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

        candidates = np.where(np.abs(sorted_mz_values - mz1) <= mz_tol)[0]
        tmp_ims = ims_vals[candidates]

        match_indices = np.where(np.abs(tmp_ims - ims1) <= ims_tol)[0]
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
    >>> get_break_indices(tmp)
    (array([0, 4, 7, 9]), array([ 1,  7, 11, 13]))
    """
    if break_values is None:
        break_values = inds
    breaks = 1 + np.where(np.diff(break_values) > min_diff)[0]
    breaks = np.concatenate([np.array([0]), breaks, np.array([inds.size - 1])])

    break_indices = inds[breaks]
    return breaks, break_indices


def _get_precursor_index_windows(
    dia_data: TimsTOF,
    precursor_index: int,
    progress: bool = True,
    mz_range: None | tuple[float, float] = None,
) -> dict[dict[list]]:
    PRELIM_N_PEAK_FILTER = 10_000  # noqa
    MIN_INTENSITY_KEEP = 50  # noqa
    MIN_NUM_SEEDS = 10_000  # noqa
    IMS_TOL = 0.01  # noqa
    # these tolerances are set narrow on purpose, since they
    # only delimit the definition of the neighborhood, which will iteratively
    # expanded.
    MZ_TOL = 0.02  # noqa
    MAX_ISOTOPE_CHARGE = 3  # noqa

    inds = dia_data[{"precursor_indices": precursor_index}, "raw"]

    breaks, break_inds = get_break_indices(inds=inds)

    push_data = dia_data.convert_from_indices(
        break_inds[:-1],
        raw_indices_sorted=True,
        return_rt_values_min=True,
        # I think this is a better value than the actual RT
        # since accounts for accumulation time.
        return_quad_mz_values=True,
        return_scan_indices=True,
    )

    pbar = tqdm(
        enumerate(zip(breaks[:-1], breaks[1:])),
        total=len(breaks),
        disable=not progress,
        desc=f"Collecting spectra for precursor={precursor_index}",
    )
    quad_splits = {}
    for bi, (startind, endind) in pbar:
        curr_push_data = {k: v[bi] for k, v in push_data.items()}
        peak_data = dia_data.convert_from_indices(
            inds[startind:endind],
            raw_indices_sorted=True,
            return_mz_values=True,
            return_corrected_intensity_values=True,
            return_mobility_values=True,
        )

        start_len = len(peak_data["mz_values"])
        # keep_intens = peak_data["corrected_intensity_values"] > MIN_INTENSITY_KEEP
        # peak_data = {k: v[keep_intens] for k, v in peak_data.items()}

        if len(peak_data["corrected_intensity_values"]) > PRELIM_N_PEAK_FILTER:
            partition_indices = np.argpartition(
                peak_data["corrected_intensity_values"],
                -PRELIM_N_PEAK_FILTER,
            )[-PRELIM_N_PEAK_FILTER:]
            peak_data = {k: v[partition_indices] for k, v in peak_data.items()}

        sort_idx = np.argsort(peak_data["mz_values"])
        peak_data = {k: v[sort_idx] for k, v in peak_data.items()}
        if len(peak_data["corrected_intensity_values"]) == 0:
            continue

        ims_values = peak_data["mobility_values"]
        mz_values = peak_data["mz_values"]
        intensity_values = peak_data["corrected_intensity_values"]

        if mz_range is not None:
            mz_filter = slice(
                np.searchsorted(mz_values, mz_range[0]),
                np.searchsorted(mz_values, mz_range[1], "right"),
            )
            ims_values = ims_values[mz_filter]
            mz_values = mz_values[mz_filter]
            intensity_values = intensity_values[mz_filter]

        f = collapse_ims(
            ims_values=ims_values,
            mz_values=mz_values,
            intensity_values=intensity_values,
            top_n=MIN_NUM_SEEDS,
            ims_tol=IMS_TOL,
            mz_tol=MZ_TOL,
        )
        filter = f["intensity"] > MIN_INTENSITY_KEEP
        f = {k: v[filter] for k, v in f.items()}

        d_out = {"mz": None, "intensity": None, "ims": None}
        if len(f["mz"]) < 0:
            new_order = np.argsort(f["mz"])
            f = {k: v[new_order] for k, v in f.items()}

        # d_out["mz"], d_out["intensity"], d_out["ims"] = deisotope_with_ims(
        #     ims_diff=IMS_TOL,
        #     ims_unit="abs",
        #     imss=f["ims"],
        #     inten=f["intensity"],
        #     mz=f["mz"],
        #     mz_unit="da",
        #     track_indices=False,
        #     mz_diff=MZ_TOL,
        #     max_charge=MAX_ISOTOPE_CHARGE,
        # )
        d_out["mz"], d_out["intensity"], d_out["ims"] = (
            f["mz"],
            f["intensity"],
            f["ims"],
        )
        pct_compression = round(len(d_out["mz"]) / start_len, ndigits=2)
        final_len = len(f["intensity"])
        pbar.set_postfix(
            {
                "peaks": start_len,
                "f_peaks": final_len,
                "pct": pct_compression,
                "min": np.min(peak_data["corrected_intensity_values"]),
                "max": np.max(peak_data["corrected_intensity_values"]),
                "f_min": int(np.min(d_out["intensity"])) if final_len else 0,
                "f_max": int(np.max(d_out["intensity"])) if final_len else 0,
            },
            refresh=False,
        )

        curr_push_data.update(d_out)
        quad_name = (
            f"{curr_push_data['quad_low_mz_values']:.6f},"
            f" {curr_push_data['quad_high_mz_values']:.6f}"
        )
        quad_splits.setdefault(quad_name, []).append(curr_push_data)
    return quad_splits


# @profile
def _collapse_seeds(
    seeds: dict[int, list[int]],
    intensity_values: NDArray[np.float32],
) -> tuple[dict[int, int], set[int]]:
    seeds = seeds.copy()
    seed_keys = list(seeds.keys())
    seed_order = np.argsort(-intensity_values[np.array(seed_keys)])
    taken = set()
    out_seeds = {}
    MAX_ITER = 5  # noqa

    for s in seed_order:
        sk = seed_keys[s]
        if sk in taken:
            continue

        out_seeds[sk] = set(seeds.pop(sk, {sk}))
        curr_len = 1

        for _ in range(MAX_ITER):
            neigh_set = set(chain(*[seeds.pop(x, set()) for x in out_seeds[sk]])).union(
                out_seeds[sk],
            )
            untaken = neigh_set.difference(taken)
            taken.update(untaken)
            out_seeds[sk].update(untaken)
            t_curr_len = len(out_seeds[sk])
            if curr_len == t_curr_len:
                break
            curr_len = t_curr_len

    out_seeds = {k: v for k, v in out_seeds.items() if len(v) > 0}
    return out_seeds, taken


# @profile
def collapse_ims(
    ims_values,
    mz_values,
    intensity_values,
    top_n=500,
    top_n_pct=0.2,
    ims_tol=0.01,
    mz_tol=0.01,
) -> dict[str, NDArray[np.float32]]:
    """Collapses peaks with similar IMS and MZ.

    Sample output
    -------------
    ```
    bundled = {
        "ims": np.zeros(..., dtype=np.float32),
        "mz": np.zeros(..., dtype=np.float32),
        "intensity": np.zeros(..., dtype=np.float32),
    }
    ```.
    """
    MIN_NEIGHBORS_SEED = 3  # noqa
    assert is_sorted(mz_values)

    neighborhoods = find_neighbors_mzsort(
        ims_vals=ims_values,
        sorted_mz_values=mz_values,
        intensities=intensity_values,
        top_n=top_n,
        top_n_pct=top_n_pct,
        ims_tol=ims_tol,
        mz_tol=mz_tol,
    )

    # unambiguous = {k for k, v in neighborhoods.items() if len(v) == 1}
    ambiguous = {k: v for k, v in neighborhoods.items() if len(v) > 1}
    seeds = {k: v for k, v in ambiguous.items() if len(v) >= MIN_NEIGHBORS_SEED}

    if seeds:
        out_seeds, taken = _collapse_seeds(
            seeds=seeds,
            intensity_values=intensity_values,
        )
    else:
        out_seeds, taken = {}, set()

    ambiguous_untaken = list(set(ambiguous).difference(taken))
    bundled = {
        "ims": np.zeros(len(out_seeds), dtype=np.float32),
        "mz": np.zeros(len(out_seeds), dtype=np.float32),
        "intensity": np.zeros(len(out_seeds), dtype=np.float32),
    }
    for i, v in enumerate(out_seeds.values()):
        v = list(v)
        v_intensities = intensity_values[v]
        # This generates the weighted average of the ims and mz
        # as the new values for the peak.
        bundled["intensity"][i] = (tot_intensity := v_intensities.sum())
        bundled["ims"][i] = (ims_values[v] * v_intensities).sum() / tot_intensity
        bundled["mz"][i] = (mz_values[v] * v_intensities).sum() / tot_intensity

    # # # TODO consider whether I really want to keep ambiduous matches
    # # # In theory I could keep only the expanded seeds.
    # unambiguous_out = {
    #     "ims": ims_values[list(chain(unambiguous, ambiguous_untaken))],
    #     "mz": mz_values[list(chain(unambiguous, ambiguous_untaken))],
    #     "intensity": intensity_values[list(chain(unambiguous, ambiguous_untaken))],
    # }

    unambiguous_out = {
        "ims": ims_values[list(ambiguous_untaken)],
        "mz": mz_values[list(ambiguous_untaken)],
        "intensity": intensity_values[list(ambiguous_untaken)],
    }
    final = {
        k: np.concatenate([unambiguous_out[k], bundled[k]]) for k in unambiguous_out
    }

    return final
