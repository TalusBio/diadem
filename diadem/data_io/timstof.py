from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from typing import Iterator, Literal

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
from diadem.deisotoping import deisotope
from diadem.search.metrics import get_ref_trace_corrs
from diadem.utils import is_sorted

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
        ims_tolerance: float = 0.1,
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
            center=index, window=window, length=len(group.mzs)
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
        u_center_mzs, inv = np.unique(center_mzs, return_inverse=True)
        u_center_intensities = np.zeros(
            len(u_center_mzs), dtype=center_intensities.dtype
        )
        np.add.at(u_center_intensities, inv, center_intensities)
        assert is_sorted(u_center_mzs)

        # After makig it unique, we deisotope the spectrum
        # after this, getting the indices that generated du_center_mzs[0]
        # would be np.where(np.isin(inv, du_center_indices[0]))

        du_center_mzs, du_center_intensities, du_center_indices = deisotope(
            u_center_mzs,
            u_center_intensities,
            max_charge=3,
            diff=mz_tolerance,
            unit=mz_tolerance_unit,
            track_indices=True,
        )
        int_keep = du_center_intensities >= (
            du_center_intensities.max() * min_intensity_ratio
        )
        du_center_mzs, du_center_intensities = (
            du_center_mzs[int_keep],
            du_center_intensities[int_keep],
        )
        assert is_sorted(du_center_mzs)

        xic_outs = []

        for i, (m, inten, ims) in enumerate(zip(mzs, intensities, imss)):
            # We first filter the peaks that are inside our IMS tolerance
            # By getting their indices.
            t_int_keep = np.abs(ims - bp_ims) <= ims_tolerance
            t_int_keep = np.where(t_int_keep)[0]

            m = m[t_int_keep]
            inten = inten[t_int_keep]

            u_mzs, inv = np.unique(m, return_inverse=True)
            u_intensities = np.zeros(len(u_mzs), dtype=inten.dtype)
            np.add.at(u_intensities, inv, inten)

            du_mzs, du_intensities, du_indices = deisotope(
                u_mzs,
                u_intensities,
                max_charge=3,
                diff=mz_tolerance,
                unit=mz_tolerance_unit,
                track_indices=True,
            )

            assert is_sorted(du_mzs)
            outs, inds = xic(
                query_mz=du_mzs,
                query_int=du_intensities,
                mzs=du_center_mzs,
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
                        [np.where(np.isin(inv, du_indices[w]))[0] for w in y]
                    )
                    out_inds.append(np.unique(t_int_keep[collapsed_indices]))
                else:
                    out_inds.append([])

            xic_outs.append((outs, out_inds))
            if i == center_index:
                assert xic_outs[-1][0].sum() >= du_center_intensities.max()

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
            du_center_mzs = du_center_mzs[keep]
            du_center_intensities = du_center_intensities[keep]
            indices = [[y for y, k in zip(x, keep) if k] for x in indices]

        ref_id = np.argmax(stacked_arr[..., center_index])
        bp_int = stacked_arr[ref_id, center_index]

        out = TimsStackedChromatograms(
            array=stacked_arr,
            mzs=du_center_mzs,
            ref_index=ref_id,
            parent_index=index,
            base_peak_intensity=bp_int,
            stack_peak_indices=indices,
            center_intensities=du_center_intensities,
            ref_ims=bp_ims,
        )
        return out


@dataclass
class TimsScanGroup(ScanGroup):
    imss: list[NDArray]

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

    def scale_window_intensities(
        self,
        index: int,
        scaling: NDArray,
        mzs: NDArray,
        window_indices: list[list[int]],
        window_mzs: NDArray,
    ) -> None:
        super().scale_window_intensities(
            index=index,
            scaling=scaling,
            mzs=mzs,
            window_indices=window_indices,
            window_mzs=window_mzs,
        )

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
        self, precursor_index: int, progress: bool = True
    ) -> dict[str:TimsScanGroup]:
        with self.lazy_datafile() as datafile:
            index_win_dicts = _get_precursor_index_windows(
                datafile, precursor_index=precursor_index, progress=progress
            )
            out = {}

            for k, v in index_win_dicts.items():
                mzs = []
                ints = []
                imss = []

                for vi in v:
                    new_order = np.argsort(vi["mz"])
                    mzs.append(vi["mz"][new_order])
                    ints.append(vi["intensity"][new_order])
                    imss.append(vi["ims"][new_order])

                # TODO change this to a data structure that stores
                # this only once.
                quad_low = list({x["quad_low_mz_values"] for x in v})
                quad_high = list({x["quad_high_mz_values"] for x in v})
                assert len(quad_high) == 1
                assert len(quad_low) == 1

                bp_indices = [np.argmax(x) if len(x) else None for x in ints]
                bp_ints = np.array(
                    [
                        x1[x2] if x2 is not None else 0
                        for x1, x2 in zip(ints, bp_indices)
                    ]
                )
                bp_mz = np.array(
                    [
                        x1[x2] if x2 is not None else -1
                        for x1, x2 in zip(mzs, bp_indices)
                    ]
                )
                bp_indices = np.array(bp_indices)
                # bp_ims = np.array([x1[x2] for x1, x2 in zip(imss, bp_indices)])
                rts = np.array([x["rt_values_min"] for x in v])
                scan_indices = [str(x["scan_indices"]) for x in v]

                x = TimsScanGroup(
                    precursor_range=(quad_low[0], quad_high[0]),
                    mzs=mzs,
                    intensities=ints,
                    imss=imss,
                    base_peak_int=bp_ints,
                    base_peak_mz=bp_mz,
                    # base_peak_ims=bp_ims,
                    iso_window_name=k,
                    retention_times=rts,
                    scan_ids=scan_indices,
                )
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
    ims_vals: NDArray[np.float64],
    sorted_mz_values: NDArray[np.float64],
    intensities: NDArray[np.float64],
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
    ims_vals: NDArray[np.float64]
        Array containing the ion mobility values of the precursor.
    sorted_mz_values: NDArray[np.float64]
        Sorted array contianing the mz values
    intensities: NDArray[np.float64]
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
        mz_tol: NDArray[np.float64] = get_tolerance(
            mz_tol, theoretical=sorted_mz_values, unit="ppm"
        )
    else:
        raise ValueError("Only 'Da' and 'ppm' values are supported as mass errors")

    top_n = int(max(len(intensities) * top_n_pct, top_n))
    if len(intensities) > top_n:
        top_indices = np.argpartition(intensities, -top_n)[-top_n:]
    else:
        intensities = None

    opts = {}
    for i1, (ims1, mz1) in enumerate(zip(ims_vals, sorted_mz_values)):
        if i1 not in top_indices:
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


def propose_boxes(
    intensities: NDArray[np.float32],
    ims_values: NDArray[np.float32],
    mz_values: NDArray[np.float64],
    ref_ims: float,
    ref_mz: float,
    mz_sizes=(0.02,),
    ims_sizes=(0.05, 0.1),
):
    """Proposes and scores bounding boxes around a peak.

    Details
    -------
    Score definition:
    The score is more of a loss than a score (since lower is better)
    The score is defined as
        [
            (max_mz_error + # max_ims_error) +
            (weighted_mz_standard_deviation) +
            # (weighted_ims_standard_deviation)
        ] / total_intensity

    Therefore, boxes with very low variance in their mzs or IMSs will have low score.
    And boxes with high intensity will also have lower intensity.

    Returns
    -------
    boxes
        Each box has 5 number, [min_mz, max_mz, min_ims, max_ims, score]
    box_intensities: list[float]
        The summed intensity in each of the boxes
    box_centroids: list[tuple[float, float]]
        Each centroid is a tuple of `mz_centroid, ims_centroid` where the
        centroid is a weighted average of that dimension, using the intensities
        as weights.

    """
    score_mutiplier = max(mz_sizes)  # + max(ims_sizes)
    delta_ims = np.abs(ims_values - ref_ims)
    delta_mzs = np.abs(mz_values - ref_mz)

    # each box has 5 number, [min_mz, max_mz, min_ims, max_ims, score]
    boxes = []
    box_centroids = []
    box_intensities = []

    for mz_tol in mz_sizes:
        for ims_tol in ims_sizes:
            box_index = (delta_ims <= ims_tol) & (delta_mzs <= mz_tol)
            tmp_mzs = mz_values[box_index]
            tmp_ims = ims_values[box_index]
            tmp_intensity = intensities[box_index]

            box_intensity = tmp_intensity.sum()
            if box_intensity == 0:
                raise RuntimeError(f"Box has 0 intensity {box_index}")
            mz_centroid = (tmp_mzs * tmp_intensity).sum() / box_intensity
            ims_centroid = (tmp_ims * tmp_intensity).sum() / box_intensity

            ims_std = (
                (delta_ims[box_index] ** 2) * tmp_intensity
            ).sum() / box_intensity
            ims_std = np.sqrt(ims_std)
            mz_std = (
                0  # ((delta_mzs[box_index] ** 2)*tmp_intensity).sum() / box_intensity
            )
            score = (score_mutiplier + (mz_std + ims_std)) / box_intensity
            boxes.append(
                [
                    ref_mz - mz_tol,
                    ref_mz + mz_tol,
                    ref_ims - ims_tol,
                    ref_ims + ims_tol,
                    score,
                ]
            )
            box_intensities.append(box_intensity)
            box_centroids.append((mz_centroid, ims_centroid))

    return boxes, box_intensities, box_centroids


# Code modified from https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh, eps=1e-5, invert_scores=False):
    # Invert_scores=True keeps lowest values for the scores (like a loss)

    # Eps adds a little edge to each side of the boxes
    # (making boxes of width or height==0 manageable)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    # Note our boxes are [min(x),max(x),min(y),max(y)], instead of [x,y,x,y] in the orifinal implementation
    x1 = boxes[:, 0] - eps
    x2 = boxes[:, 1] + eps
    y1 = boxes[:, 2] - eps
    y2 = boxes[:, 3] + eps
    scores = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)
    if invert_scores:
        idxs = idxs[::-1]

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + eps)
        h = np.maximum(0, yy2 - yy1 + eps)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        box_inds_delete = np.where(overlap > overlapThresh)[0]
        idxs = np.delete(idxs, np.concatenate(([last], box_inds_delete)))

    # return the indices of the best bounding boxes
    return pick


def bundle_neighbors_mzsorted(
    ims_values: NDArray[np.float32],
    sorted_mz_values: NDArray[np.float32],
    intensities: NDArray[np.float32],
    top_n: int = 500,
):
    """

    WARNING:
    Output is not necessarily sorted!!!
    """
    BOX_EPS = 1e-7

    opts = find_neighbors_mzsort(
        ims_values,
        sorted_mz_values=sorted_mz_values,
        intensities=intensities,
        top_n=top_n,
        ims_tol=0.1,
        mz_tol=50,
        mz_tol_unit="ppm",
    )

    unambiguous = {k for k, v in opts.items() if len(v) == 1}
    ambiguous = {k: v for k, v in opts.items() if len(v) > 1}

    unambiguous = {
        "ims": ims_values[list(unambiguous)],
        "mz": sorted_mz_values[list(unambiguous)],
        "intensity": intensities[list(unambiguous)],
    }

    if len(ambiguous) > 0:
        all_boxes = []
        all_intensities = []
        all_centroids = []
        for k, v in ambiguous.items():
            boxes, box_intensities, box_centroids = propose_boxes(
                intensities[v],
                ims_values[v],
                mz_values=sorted_mz_values[v],
                ref_ims=ims_values[k],
                ref_mz=sorted_mz_values[k],
                mz_sizes=(
                    0.01,
                    0.05,
                ),
                ims_sizes=(0.01, 0.05, 0.1),
                # ims_sizes = (0.005, 0.01, 0.02),
            )
            all_boxes.extend(boxes)
            all_intensities.extend(box_intensities)
            all_centroids.extend(box_centroids)

        best_boxes_indices = non_max_suppression_fast(
            np.stack(all_boxes), 0, eps=BOX_EPS, invert_scores=True
        )
        best_centroids = [all_centroids[x] for x in best_boxes_indices]
        best_intensities = [all_intensities[x] for x in best_boxes_indices]
        mzs, imss = zip(*best_centroids)
        boxed = {
            "ims": imss,
            "mz": mzs,
            "intensity": best_intensities,
        }

        final = {k: np.concatenate([unambiguous[k], boxed[k]]) for k in unambiguous}
    else:
        final = unambiguous
    return final


def get_break_indices(
    inds: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Gets the incides and break values for an increasing array.

    Example:
    -------
    >>> tmp = np.array([1,2,3,4,7,8,9,11,12,13])
    >>> get_break_indices(tmp)
    (array([0, 4, 7, 9]), array([ 1,  7, 11, 13]))
    """
    breaks = 1 + np.where(np.diff(inds) > 1)[0]
    breaks = np.concatenate([np.array([0]), breaks, np.array([inds.size - 1])])

    break_indices = inds[breaks]
    return breaks, break_indices


def _get_precursor_index_windows(
    dia_data: TimsTOF, precursor_index: int, progress: bool = True
) -> dict[dict[list]]:
    PRELIM_N_PEAK_FILTER = 10_000  # noqa
    MIN_INTENSITY_KEEP = 100  # noqa
    MIN_NUM_SEED_BOXES = 1000  # noqa

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
                peak_data["corrected_intensity_values"], -PRELIM_N_PEAK_FILTER
            )[-PRELIM_N_PEAK_FILTER:]
            peak_data = {k: v[partition_indices] for k, v in peak_data.items()}

        sort_idx = np.argsort(peak_data["mz_values"])
        peak_data = {k: v[sort_idx] for k, v in peak_data.items()}
        if len(peak_data["corrected_intensity_values"]) == 0:
            continue

        f = collapse_ims(
            ims_values=peak_data["mobility_values"],
            mz_values=peak_data["mz_values"],
            intensity_values=peak_data["corrected_intensity_values"],
            top_n=MIN_NUM_SEED_BOXES,
        )
        filter = f["intensity"] > MIN_INTENSITY_KEEP
        f = {k: v[filter] for k, v in f.items()}
        pct_compression = round(len(f["mz"]) / start_len, ndigits=2)
        final_len = len(f["intensity"])
        pbar.set_postfix(
            {
                "peaks": start_len,
                "f_peaks": final_len,
                "pct": pct_compression,
                "min": np.min(peak_data["corrected_intensity_values"]),
                "max": np.max(peak_data["corrected_intensity_values"]),
                "f_min": int(np.min(f["intensity"])) if final_len else 0,
                "f_max": int(np.max(f["intensity"])) if final_len else 0,
            },
            refresh=False,
        )

        curr_push_data.update(f)
        quad_name = (
            f"{curr_push_data['quad_low_mz_values']:.6f},"
            f" {curr_push_data['quad_high_mz_values']:.6f}"
        )
        quad_splits.setdefault(quad_name, []).append(curr_push_data)
    return quad_splits


# @profile
def collapse_seeds(
    seeds: dict[int, list[int]], intensity_values: NDArray[np.float64]
) -> tuple[dict[int, int], set[int]]:
    seeds = seeds.copy()
    seed_keys = list(seeds.keys())
    seed_order = np.argsort(-intensity_values[np.array(seed_keys)])
    taken = set()
    out_seeds = {}
    max_iter = 3

    for s in seed_order:
        sk = seed_keys[s]
        if sk in taken:
            continue

        out_seeds[sk] = set(seeds.pop(sk, {sk}))
        curr_len = 1

        for _ in range(max_iter):
            neigh_set = set(chain(*[seeds.pop(x, set()) for x in out_seeds[sk]])).union(
                out_seeds[sk]
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
) -> dict[str, NDArray[np.float64]]:
    neighborhoods = find_neighbors_mzsort(
        ims_vals=ims_values,
        sorted_mz_values=mz_values,
        intensities=intensity_values,
        top_n=top_n,
        top_n_pct=top_n_pct,
        ims_tol=ims_tol,
        mz_tol=mz_tol,
    )

    unambiguous = {k for k, v in neighborhoods.items() if len(v) == 1}
    ambiguous = {k: v for k, v in neighborhoods.items() if len(v) > 1}
    seeds = {k: v for k, v in ambiguous.items() if len(v) > 5}

    if seeds:
        out_seeds, taken = collapse_seeds(
            seeds=seeds, intensity_values=intensity_values
        )
    else:
        out_seeds, taken = {}, set()

    ambiguous_untaken = list(set(ambiguous).difference(taken))
    unambiguous_out = {
        "ims": ims_values[list(chain(unambiguous, ambiguous_untaken))],
        "mz": mz_values[list(chain(unambiguous, ambiguous_untaken))],
        "intensity": intensity_values[list(chain(unambiguous, ambiguous_untaken))],
    }

    bundled = {
        "ims": np.zeros(len(out_seeds), dtype=np.float64),
        "mz": np.zeros(len(out_seeds), dtype=np.float64),
        "intensity": np.zeros(len(out_seeds), dtype=np.float32),
    }
    for i, v in enumerate(out_seeds.values()):
        v = list(v)
        v_intensities = intensity_values[v]
        bundled["intensity"][i] = (tot_intensity := v_intensities.sum())
        bundled["ims"][i] = (ims_values[v] * v_intensities).sum() / tot_intensity
        bundled["mz"][i] = (mz_values[v] * v_intensities).sum() / tot_intensity

    final = {
        k: np.concatenate([unambiguous_out[k], bundled[k]]) for k in unambiguous_out
    }
    return final


if __name__ == "__main__":
    file = "/Users/sebastianpaez/git/diadem/profiling/profiling_data/LFQ_timsTOFPro_diaPASEF_Ecoli_01.d"
    config = DiademConfig()

    data = TimsSpectrumStacker(file, config)
    group = next(data.yield_iso_window_groups())
    group.get_highest_window(
        21,
        0.01,
        mz_tolerance=0.02,
        mz_tolerance_unit="da",
        ims_tolerance=0.03,
        ims_tolerance_unit="abs",
        min_correlation=0.5,
        max_peaks=150,
    )
