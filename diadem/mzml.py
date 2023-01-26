from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import ms_deisotope
import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from ms2ml import Config, Spectrum
from ms2ml.data.adapters import MZMLAdapter
from ms2ml.utils.mz_utils import annotate_peaks
from ms_deisotope.deconvolution.utils import prepare_peaklist
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm.auto import tqdm

from diadem.config import MassError
from diadem.search.metrics import get_ref_trace_corrs

try:
    zip([], [], strict=True)

    def strictzip(*args: Iterable) -> Iterable:
        """Like zip but checks that the length of all elements is the same."""
        return zip(*args, strict=True)

except TypeError:

    def strictzip(*args: Iterable) -> Iterable:
        """Like zip but checks that the length of all elements is the same."""
        args = [list(arg) for arg in args]
        lengs = {len(x) for x in args}
        if len(lengs) > 1:
            raise ValueError("All arguments need to have the same legnths")
        return zip(*args)


@dataclass
class ScanGroup:
    """Represents all spectra that share an isolation window."""

    precursor_range: tuple[float, float]
    mzs: list[NDArray]
    intensities: list[NDArray]
    base_peak_mz: NDArray[np.float32]
    base_peak_int: NDArray[np.float32]
    retention_times: NDArray
    scan_ids: list[str]
    iso_window_name: str

    def __post_init__(self) -> None:
        """Check that all the arrays have the same length."""
        elems = [
            self.mzs,
            self.intensities,
            self.base_peak_int,
            self.base_peak_mz,
            self.retention_times,
            self.scan_ids,
        ]

        # TODO move this to assertions so they can be skipped
        # during runtime
        lengths = {len(x) for x in elems}
        if len(lengths) != 1:
            raise ValueError("Not all lengths are the same")
        if len(self.precursor_range) != 2:
            raise ValueError(
                "Precursor mass range should have 2 elements,"
                f" has {len(self.precursor_range)}"
            )

    # @profile
    def get_highest_window(
        self,
        window: int,
        min_intensity_ratio: float,
        tolerance: float,
        tolerance_unit: str,
        min_correlation: float,
    ) -> StackedChromatograms:
        """Gets the highest intensity window of the chromatogram.

        Briefly ...
        1. Gets the highes peak accross all spectra in the chromatogram range.
        2. Finds what peaks are in that same spectrum.
        3. Looks for spectra around that spectrum.
        4. extracts the chromatogram for all mzs in the "parent spectrum"

        """
        top_index = np.argmax(self.base_peak_int)
        window = StackedChromatograms.from_group(
            self,
            top_index,
            window=window,
            min_intensity_ratio=min_intensity_ratio,
            min_correlation=min_correlation,
            tolerance=tolerance,
            tolerance_unit=tolerance_unit,
        )

        return window

    # TODO make this just take the stacked chromatogram object
    def scale_window_intensities(
        self,
        index: int,
        scaling: NDArray,
        mzs: NDArray,
        window_indices: list[list[int]],
        window_mzs: NDArray,
    ) -> None:
        """Scales the intensities of specific mzs in a window of the chromatogram.

        Parameters
        ----------
        index : int
            The index of the center spectrum for the the window to scale.
        scaling : NDArray
            The scaling factors to apply to the intensities. Size should be
            the same as the length of the window.
        mzs : NDArray
            The m/z values of the peaks to scale.
        window_indices : list[list[int]]
            The indices of the peaks in the window to scale. These are tracked
            internally during the workflow.
        window_mzs : NDArray
            The m/z values of the peaks in the window to scale. These are tracked
            internally during the workflow.
        """
        window = len(scaling)
        slc, center_index = slice_from_center(
            center=index, window=window, length=len(self)
        )

        # TODO this can be tracked internally ...
        match_obs_mz_indices, match_win_mz_indices = annotate_peaks(
            theo_mz=mzs,
            mz=window_mzs,
            tolerance=0.02,
            unit="da",
        )

        match_win_mz_indices = np.unique(match_win_mz_indices)

        zipped = strictzip(range(*slc.indices(len(self))), scaling, window_indices)
        for i, s, si in zipped:
            for mz_i in match_win_mz_indices:
                sim = si[mz_i]
                if len(sim) > 0:
                    self.intensities[i][sim] = self.intensities[i][sim] * s
                else:
                    continue
            self.base_peak_int[i] = np.max(self.intensities[i])
            self.base_peak_mz[i] = self.mzs[i][np.argmax(self.intensities[i])]

    def __len__(self) -> int:
        """Returns the number of spectra in the group."""
        return len(self.intensities)

    """
    @staticmethod
    def scale_matching_intensities(
        mzs,
        intensities,
        reference_mzs,
        scaling_factor,
        tolerance: float,
        tolerance_unit: MassError,
    ):
        # TODO optimize this
        indices_1, indices_2 = annotate_peaks(
            theo_mz=mzs,
            mz=reference_mzs,
            tolerance=tolerance,
            unit=tolerance_unit,
        )

        indices_1 = np.unique(indices_1)

        if len(indices_1) == 0:
            return intensities, True

        vals = intensities[indices_1]
        # min_val = vals.min()
        # max_val = vals.max()
        # intensities[indices_1] = np.clip(
        #     vals * scaling_factor, min_val, max_val * scaling_factor
        # )
        if (scaling_factor > 1) or (scaling_factor < 0):
            raise ValueError("Scaling factor should be a number between 0 and 1")
        intensities[indices_1] = vals * scaling_factor
        return intensities, False
    """


# @profile
def xic(
    query_mz: NDArray[np.float32],
    query_int: NDArray[np.float32],
    mzs: NDArray[np.float32],
    tolerance_unit: MassError = "da",
    tolerance: float = 0.02,
) -> NDArray[np.float32]:
    """Gets the extracted ion chromatogram form arrays.

    Gets the extracted ion chromatogram from the passed mzs and intensities
    The output should be the same length as the passed mzs.

    """
    theo_mz_indices, obs_mz_indices = annotate_peaks(
        theo_mz=mzs,
        mz=query_mz,
        tolerance=tolerance,
        unit=tolerance_unit,
    )

    outs = np.zeros_like(mzs, dtype="float")
    inds = []
    for i in range(len(outs)):
        query_indices = obs_mz_indices[theo_mz_indices == i]
        ints_subset = query_int[query_indices]
        if len(ints_subset) == 0:
            inds.append([])
        else:
            outs[i] = np.sum(ints_subset)
            inds.append(query_indices)

    return outs, inds


def slice_from_center(center: int, window: int, length: int) -> tuple[slice, int]:
    """Generates a slice provided a center and window size.

    Creates a slice that accounts for the endings of an iterable
    in such way that the window size is maintained.

    Examples
    --------
    >>> my_list = [0,1,2,3,4,5,6]
    >>> slc, center_index = slice_from_center(
    ...     center=4, window=3, length=len(my_list))
    >>> slc
    slice(3, 6, None)
    >>> my_list[slc]
    [3, 4, 5]
    >>> my_list[slc][center_index] == my_list[4]
    True

    >>> slc = slice_from_center(1, 3, len(my_list))
    >>> slc
    (slice(0, 3, None), 1)
    >>> my_list[slc[0]]
    [0, 1, 2]

    >>> slc = slice_from_center(6, 3, len(my_list))
    >>> slc
    (slice(4, 7, None), 2)
    >>> my_list[slc[0]]
    [4, 5, 6]
    >>> my_list[slc[0]][slc[1]] == my_list[6]
    True

    """
    start = center - (window // 2)
    end = center + (window // 2) + 1
    center_index = window // 2

    if start < 0:
        start = 0
        end = window
        center_index = center

    if end >= length:
        end = length
        start = end - window
        center_index = window - (length - center)

    slice_q = slice(start, end)
    return slice_q, center_index


@dataclass
class StackedChromatograms:
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
    parent_index :
        Identifier of the range where the window was extracted
    base_peak_intensity :
        Intensity of the base peak in the reference spectrum
    stack_peak_indices :
        List of indices used to stack the array, it is a list of dimensions [w]

    Details
    -------
    The dimensions of the arrays are `w` the window
    size of the extracted ion chromatogram. `i` the number
    of m/z peaks that were extracted.

    """

    array: NDArray[np.float32]
    mzs: NDArray[np.float32]
    ref_index: int
    parent_index: int
    base_peak_intensity: float
    stack_peak_indices: list[list[int]] | list[NDArray[np.int32]]
    center_intensities: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Checks that the dimensions of the arrays are correct.

        Since they are assertions, they are not meant to be needed for the
        correct working of the
        """
        array_i = self.array.shape[-2]
        array_w = self.array.shape[-1]

        mz_i = self.mzs.shape[-1]

        assert (
            self.ref_index <= mz_i
        ), f"Reference index outside of mz values {self.ref_index} > {mz_i}"
        assert (
            array_i == mz_i
        ), f"Intensity Array and mzs have different lengths {array_i} != {mz_i}"
        for i, x in enumerate(self.stack_peak_indices):
            assert len(x) == mz_i, (
                f"Number of mzs and number of indices {len(x)} != {mz_i} is different"
                f" for {i}"
            )
        assert array_w == len(
            self.stack_peak_indices
        ), "Window size is not respected in the stack"

    @property
    def ref_trace(self) -> NDArray[np.float32]:
        """Returns the reference trace.

        The reference trace is the extracted ion chromatogram of the
        mz that corresponds to the highest intensity peak.
        """
        return self.array[self.ref_index, ...]

    @property
    def ref_mz(self) -> float:
        """Returns the m/z value of the reference trace."""
        return self.mzs[self.ref_index]

    @property
    def ref_fwhm(self) -> int:
        """Returns the number of points in the reference trace above half max.

        Not really fwhm, just number of elements above half max.
        """
        rt = self.ref_trace
        rt = rt - rt.min()
        above_hm = rt >= (rt.max() / 2)
        return above_hm.astype(int).sum()

    def plot(self, plt) -> None:  # noqa
        """Plots the stacked chromatogram as lines."""
        # TODO reconsider this implementation, maybe lazy import
        # of matplotlib.
        plt.plot(self.array.T)
        plt.plot(self.array[self.ref_index, ...].T, color="black")
        plt.show()

    def trace_correlation(self) -> NDArray[np.float32]:
        """Calculate the correlation between the reference trace and all other traces.

        Returns
        -------
        NDArray[np.float32]
            An array of shape [i] where i is the number of traces
            in the stacked chromatogram.
        """
        return get_ref_trace_corrs(arr=self.array, ref_idx=self.ref_index)

    @staticmethod
    # @profile
    def from_group(
        group: ScanGroup,
        index: int,
        window: int = 21,
        tolerance: float = 0.02,
        tolerance_unit: MassError = "da",
        min_intensity_ratio: float = 0.01,
        min_correlation: float = 0.5,
    ) -> StackedChromatograms:
        """Create a stacked chromatogram from a scan group.

        Parameters
        ----------
        group : ScanGroup
            A scan group containing the spectra to stack
        index : int
            The index of the spectrum to use as the reference
        window : int, optional
            The number of spectra to stack, by default 21
        tolerance : float, optional
            The tolerance to use when matching m/z values, by default 0.02
        tolerance_unit : MassError, optional
            The unit of the tolerance, by default "da"
        min_intensity_ratio : float, optional
            The minimum intensity ratio to use when stacking, by default 0.01
        min_correlation : float, optional
            The minimum correlation to use when stacking, by default 0.5

        """
        # The center index is the same as the provided index
        # Except in cases where the edge of the group is reached, where
        # the center index is adjusted to the edge of the group
        slice_q, center_index = slice_from_center(
            center=index, window=window, length=len(group.mzs)
        )
        mzs = group.mzs[slice_q]
        intensities = group.intensities[slice_q]

        center_mzs = mzs[center_index]
        center_intensities = intensities[center_index]

        int_keep = center_intensities >= (
            center_intensities.max() * min_intensity_ratio
        )

        # num_keep = int_keep.sum()
        # logger.debug("Number of peaks to stack: "
        # f"{len(center_mzs)}, number above 0.1% intensity {num_keep} "
        # f"[{100*num_keep/len(center_mzs):.02f} %]")
        center_mzs = center_mzs[int_keep]
        center_intensities = center_intensities[int_keep]

        xic_outs = []

        for i, (m, inten) in enumerate(zip(mzs, intensities)):
            xic_outs.append(
                xic(
                    query_mz=m,
                    query_int=inten,
                    mzs=center_mzs,
                    tolerance=tolerance,
                    tolerance_unit=tolerance_unit,
                )
            )
            if i == center_index:
                assert xic_outs[-1][0].sum() >= center_intensities.max()

        stacked_arr = np.stack([x[0] for x in xic_outs], axis=-1)
        indices = [x[1] for x in xic_outs]

        if min_correlation and stacked_arr.shape[-2] > 1:
            ref_id = np.argmax(stacked_arr[..., center_index])
            corrs = get_ref_trace_corrs(arr=stacked_arr, ref_idx=ref_id)

            # I think adding the 1e-5 is needed here due to numric instability
            # in the flaoting point operation
            assert np.max(corrs) <= (
                corrs[ref_id] + 1e-5
            ), "Reference does not have max corrr"

            keep = corrs >= min_correlation
            stacked_arr = stacked_arr[..., keep, ::1]
            center_mzs = center_mzs[keep]
            center_intensities = center_intensities[keep]
            indices = [[y for y, k in zip(x, keep) if k] for x in indices]

        ref_id = np.argmax(stacked_arr[..., center_index])
        bp_int = stacked_arr[ref_id, center_index]

        out = StackedChromatograms(
            array=stacked_arr,
            mzs=center_mzs,
            ref_index=ref_id,
            parent_index=index,
            base_peak_intensity=bp_int,
            stack_peak_indices=indices,
            center_intensities=center_intensities,
        )
        return out


class SpectrumStacker:
    """Helper class that stacks the spectra of an mzml file into chromatograms."""

    def __init__(self, mzml_file: Path | str, config: Config) -> None:
        """Initializes the SpectrumStacker class.

        Parameters
        ----------
        mzml_file : Path | str
            Path to the mzml file.
        config : DiademConfig
            The configuration object. Note that this is an DiademConfig
            configuration object.
        """
        self.adapter = MZMLAdapter(mzml_file, config=config)

        # TODO check if directly reading the xml is faster ...
        # also evaluate if that is needed
        scaninfo = self.adapter.get_scan_info()
        self.config = config
        self.ms2info = scaninfo[scaninfo.ms_level > 1].copy().reset_index()
        self.unique_iso_windows = set(np.array(self.ms2info.iso_window))

    def _get_iso_window_group(
        self, iso_window_name: str, iso_window: tuple[float, float], chunk: DataFrame
    ) -> ScanGroup:
        logger.debug(f"Processing iso window {iso_window_name}")

        window_mzs = []
        window_ints = []
        window_bp_mz = []
        window_bp_int = []
        window_rtinsecs = []
        window_scanids = []

        for row in tqdm(
            chunk.itertuples(), desc=f"Preprocessing spectra for {iso_window_name}"
        ):
            spec_id = row.spec_id
            curr_spec: Spectrum = self.adapter[spec_id]
            # NOTE instrument seems to have a wrong value ...
            # Also activation seems to not be recorded ...
            curr_spec = curr_spec.filter_top(self.config.run_max_peaks_per_spec)

            # Deisotoping!
            if self.config.run_deconvolute_spectra:
                peaks = prepare_peaklist((curr_spec.mz, curr_spec.intensity))
                deconvoluted_peaks, _ = ms_deisotope.deconvolute_peaks(
                    peaks,
                    averagine=ms_deisotope.peptide,
                    scorer=ms_deisotope.MSDeconVFitter(0),
                    retention_strategy=ms_deisotope.deconvolution.TopNRetentionStrategy(
                        50, max_mass=2000
                    ),
                    charge_range=(1, 3),
                )

                # For scorer discussion please refer to
                # check https://mobiusklein.github.io/ms_deisotope/docs/_build/html/deconvolution/envelope_scoring.html#ms_deisotope.scoring.IsotopicFitterBase

                mzs = np.array([x.mz for x in deconvoluted_peaks])
                intensities = np.array([x.intensity for x in deconvoluted_peaks])
            else:
                mzs = curr_spec.mz
                intensities = curr_spec.intensity
            # TODO evaluate this scaling
            intensities = np.sqrt(intensities)
            if len(mzs) == 0:
                mzs, intensities = np.array([0]), np.array([0])
            bp_index = np.argmax(intensities)
            bp_mz = mzs[bp_index]
            bp_int = intensities[bp_index]
            rtinsecs = curr_spec.retention_time.seconds()

            window_mzs.append(mzs)
            window_ints.append(intensities)
            window_bp_mz.append(bp_mz)
            window_bp_int.append(bp_int)
            window_rtinsecs.append(rtinsecs)
            window_scanids.append(spec_id)

        # Create datasets within each group
        logger.info(f"Saving group {iso_window_name} with length {len(window_mzs)}")

        window_bp_mz = np.array(window_bp_mz).astype(np.float32)
        window_bp_int = np.array(window_bp_int).astype(np.float32)
        window_rtinsecs = np.array(window_rtinsecs).astype(np.float16)
        window_scanids = np.array(window_scanids, dtype="object")

        group = ScanGroup(
            iso_window_name=iso_window_name,
            precursor_range=iso_window,
            mzs=window_mzs,
            intensities=window_ints,
            base_peak_mz=window_bp_mz,
            base_peak_int=window_bp_int,
            retention_times=window_rtinsecs,
            scan_ids=window_scanids,
        )
        return group

    def get_iso_window_groups(
        self, workerpool: None | Parallel = None
    ) -> list[ScanGroup]:
        """Returns a list of all ScanGroups in an mzML file."""
        grouped = self.ms2info.sort_values("RTinSeconds").groupby("iso_window")
        iso_windows, chunks = zip(*list(grouped))

        # logger.error(
        #     "Sebastian has not removed this from the code! do not let this go though!"
        # )
        # iso_windows, chunks = zip(
        #     *[
        #         (iso_window, chunk)
        #         for iso_window, chunk in zip(iso_windows, chunks)
        #         if iso_window[0] > 400 and iso_window[0] < 420
        #     ]
        # )
        iso_window_names = [
            "({:.06f}, {:.06f})".format(*iso_window) for iso_window in iso_windows
        ]

        if workerpool is None:
            results = [
                self._get_iso_window_group(
                    iso_window_name=iwn, iso_window=iw, chunk=chunk
                )
                for iwn, iw, chunk in zip(iso_window_names, iso_windows, chunks)
            ]
        else:
            results = workerpool(
                delayed(self._get_iso_window_group)(
                    iso_window_name=iwn, iso_window=iw, chunk=chunk
                )
                for iwn, iw, chunk in zip(iso_window_names, iso_windows, chunks)
            )

        return results

    def yield_iso_window_groups(self, progress: bool = False) -> Iterator[ScanGroup]:
        """Yield scan groups for each unique isolation window."""
        grouped = self.ms2info.sort_values("RTinSeconds").groupby("iso_window")

        for i, (iso_window, chunk) in enumerate(
            tqdm(grouped, disable=not progress, desc="Unique Isolation Windows")
        ):
            # Leaving here during early development and benchmarking
            # TODO delete this
            iso_window_name = "({:.06f}, {:.06f})".format(*iso_window)
            if iso_window[0] < 700 or iso_window[0] > 750:
                logger.error(
                    f"Skipping scans {iso_window_name} not in in the 700-750 range for"
                    " a debug run!"
                )
                continue

            group = self._get_iso_window_group(
                iso_window_name=iso_window_name, iso_window=iso_window, chunk=chunk
            )
            yield group
