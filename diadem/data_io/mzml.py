from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from ms2ml import Spectrum
from ms2ml.data.adapters import MZMLAdapter
from numpy.typing import NDArray
from tqdm.auto import tqdm

from diadem.config import DiademConfig, MassError
from diadem.data_io.utils import slice_from_center, strictzip, xic
from diadem.deisotoping import deisotope
from diadem.search.metrics import get_ref_trace_corrs
from diadem.utilities.utils import check_sorted, plot_to_log

# TODO re-make some of these classes as ABCs
# It would make intent explicit, since they are sub-classed
# for timstof data equivalents.


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

    precursor_mzs: list[NDArray]
    precursor_intensities: list[NDArray]
    precursor_retention_times: NDArray

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
                (
                    "Precursor mass range should have 2 elements,"
                    f" has {len(self.precursor_range)}"
                ),
            )
        plot_to_log(
            self.base_peak_int,
            title=f"Base peak chromatogram for the Group in {self.iso_window_name}",
        )

    @property
    def cache_file_stem(self) -> str:
        stem = "".join(x if x.isalnum() else "_" for x in self.iso_window_name)
        return stem

    def to_cache(self, dir: Path) -> None:
        """Saves the group to a cache file."""
        Path(dir).mkdir(parents=True, exist_ok=True)
        fragment_df = self.as_dataframe()
        precursor_df = self.precursor_dataframe()

        stem = self.cache_file_stem

        precursor_df.write_parquet(dir / f"{stem}_precursors.parquet")
        fragment_df.write_parquet(dir / f"{stem}_fragments.parquet")

    @classmethod
    def _elems_from_fragment_cache(cls, file):
        fragment_data = pl.read_parquet(file).to_dict()
        precursor_range = (
            fragment_data.pop("precursor_start")[0],
            fragment_data.pop("precursor_end")[0],
        )
        ind_max_int = [np.argmax(x) for x in fragment_data["intensities"]]
        base_peak_mz = np.array(
            [x[i] for x, i in zip(fragment_data["mzs"], ind_max_int)],
        )
        base_peak_int = np.array(
            [x[i] for x, i in zip(fragment_data["intensities"], ind_max_int)],
        )

        out = {
            "precursor_range": precursor_range,
            "mzs": fragment_data["mzs"],
            "intensities": fragment_data["intensities"],
            "base_peak_mz": base_peak_mz,
            "base_peak_int": base_peak_int,
            "retention_times": fragment_data["retention_times"],
            "scan_ids": fragment_data["scan_ids"],
        }

        return out, fragment_data

    def _precursor_elems_from_cache(self, file):
        precursor_data = pl.read_parquet(file).to_dict()
        out = {
            "precursor_mzs": precursor_data["precursor_mzs"],
            "precursor_intensities": precursor_data["precursor_intensities"],
            "precursor_retention_times": precursor_data["precursor_retention_times"],
        }
        return out, precursor_data

    @classmethod
    def from_cache(cls, dir: Path, name: str) -> ScanGroup:
        """Loads a group from a cache file."""
        fragment_elems, _fragment_data = cls._elems_from_fragment_cache(
            dir / f"{name}_fragments.parquet",
        )
        precursor_elems, _fragment_data = cls._precursor_elems_from_cache(
            dir / f"{name}_precursors.parquet",
        )
        return cls(
            iso_window_name=name,
            **precursor_elems,
            **fragment_elems,
        )

    def as_dataframe(self) -> pl.DataFrame:
        """Returns a dataframe with the data in the group.

        The dataframe has the following columns:
        - mzs: list of mzs for each spectrum
        - intensities: list of intensities for each spectrum
        - retention_times: retention times for each spectrum
        - precursor_start: start of the precursor range
        - precursor_end: end of the precursor range
        """
        out = pl.DataFrame(
            {
                "mzs": self.mzs,
                "intensities": self.intensities,
                "retention_times": self.retention_times,
                "scan_ids": self.scan_ids,
            },
        )
        out["precursor_start"] = min(self.precursor_range)
        out["precursor_end"] = max(self.precursor_range)
        return out

    def precursor_dataframe(self) -> pl.DataFrame:
        """Returns a dataframe with the metadata for the group.

        The dataframe has the following columns:
        - precursor_mzs: list of precursor mzs for each spectrum
        - precursor_intensities: list of precursor intensities for each spectrum
        - precursor_retention_times: precursor retention times for each spectrum
        - precursor_start: start of the precursor range
        - precursor_end: end of the precursor range
        """
        out = pl.DataFrame(
            {
                "precursor_mzs": self.precursor_mzs,
                "precursor_intensities": self.precursor_intensities,
                "precursor_retention_times": self.precursor_retention_times,
            },
        )
        return out

    def get_highest_window(
        self,
        window: int,
        min_intensity_ratio: float,
        mz_tolerance: float,
        mz_tolerance_unit: str,
        min_correlation: float,
        max_peaks: int,
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
            group=self,
            index=top_index,
            window=window,
            min_intensity_ratio=min_intensity_ratio,
            min_correlation=min_correlation,
            mz_tolerance=mz_tolerance,
            mz_tolerance_unit=mz_tolerance_unit,
            max_peaks=max_peaks,
        )

        return window

    # TODO make this just take the stacked chromatogram object
    def scale_window_intensities(
        self,
        index: int,
        scaling: NDArray,
        window_indices: list[list[int]],
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
            center=index,
            window=window,
            length=len(self),
        )
        slc = range(*slc.indices(len(self)))

        zipped = strictzip(slc, scaling, window_indices)
        for i, s, si in zipped:
            inds = [np.array(x) for x in si if len(x)]
            if inds:
                inds = np.unique(np.concatenate(inds))
                self._scale_spectrum_at(
                    spectrum_index=i,
                    value_indices=inds,
                    scaling_factor=s,
                )

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

        if len(self.intensities[i]):
            self.base_peak_int[i] = np.max(self.intensities[i])
            self.base_peak_mz[i] = self.mzs[i][np.argmax(self.intensities[i])]
        else:
            self.base_peak_int[i] = -1
            self.base_peak_mz[i] = -1

    def get_precursor_evidence(
        self,
        rt: float,
        mzs: NDArray[np.float32],
        mz_tolerance: float,
        mz_tolerance_unit: Literal["ppm", "Da"] = "ppm",
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Finds precursor information for a given RT and m/z.

        NOTE: This is a first implementation of the functionality,
        therefore it is very simple and prone to optimization and
        rework.

        1. Find the closest RT.
        2. Find if there are peaks that match the mzs.
        3. Return a list of dm and a list of intensities for each.

        Parameters
        ----------
        rt : float
            The retention time to find precursor information for.
        mzs : NDArray[np.float32]
            The m/z values to find precursor information for.
        mz_tolerance : float
            The m/z tolerance to use when finding precursor information.
        mz_tolerance_unit : str, optional
            The unit of the m/z tolerance, by default "ppm"

        Returns
        -------
        tuple[NDArray[np.float32], list[NDArray[np.float32]]]]
            A array with the list of intensities and a list arrays,
            each of which is the for the values integrated for the
            intensity values.
        """
        index = np.searchsorted(self.precursor_retention_times, rt)
        slc, center_index = slice_from_center(
            index,
            window=11,
            length=len(self.precursor_mzs),
        )
        q_mzs = self.precursor_mzs[index]

        q_intensities = self.precursor_intensities[index]
        # TODO change preprocessing of the MS1 level to make it more
        # permissive, cleaner spectra is not critical here.

        out_ints = []
        out_dms = []

        if len(q_intensities) > 0:
            for q_mzs, q_intensities in zip(
                self.precursor_mzs[slc],
                self.precursor_intensities[slc],
            ):
                intensities, indices = xic(
                    query_mz=q_mzs,
                    query_int=q_intensities,
                    mzs=mzs,
                    tolerance=mz_tolerance,
                    tolerance_unit=mz_tolerance_unit,
                )
                dms = [q_mzs[inds] - match_mz for inds, match_mz in zip(indices, mzs)]
                out_ints.append(intensities)
                out_dms.append(dms)

            intensities = np.stack(out_ints, axis=0).sum(axis=0)
            dms = out_dms[center_index]
        else:
            intensities = np.zeros_like(mzs)
            dms = [[] for _ in range(len(mzs))]

        return intensities, dms

    def __len__(self) -> int:
        """Returns the number of spectra in the group."""
        return len(self.intensities)


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
        List of indices used to stack the array, it is a list of dimensions [w],
        where each element can be either a list of indices or an empty list.

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
            self.stack_peak_indices,
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
        mz_tolerance: float = 0.02,
        mz_tolerance_unit: MassError = "da",
        min_intensity_ratio: float = 0.01,
        min_correlation: float = 0.5,
        max_peaks: int = 150,
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
        mz_tolerance : float, optional
            The tolerance to use when matching m/z values, by default 0.02
        mz_tolerance_unit : MassError, optional
            The unit of the tolerance, by default "da"
        min_intensity_ratio : float, optional
            The minimum intensity ratio to use when stacking, by default 0.01
        min_correlation : float, optional
            The minimum correlation to use when stacking, by default 0.5
        max_peaks : int, optional
            The maximum number of peaks to return in a group, by default is 150.
            If the candidates is more than this number, it will the best co-eluting
            peaks.

        """
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
                    tolerance=mz_tolerance,
                    tolerance_unit=mz_tolerance_unit,
                ),
            )
            if i == center_index:
                assert xic_outs[-1][0].sum() >= center_intensities.max()

        stacked_arr = np.stack([x[0] for x in xic_outs], axis=-1)

        # TODO make this an array and subset it in line 457
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

    def __init__(self, mzml_file: Path | str, config: DiademConfig) -> None:
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
        scaninfo = pl.DataFrame(self.adapter.get_scan_info()).fill_null(0.0)
        ms1_scaninfo = scaninfo.filter(pl.col("ms_level") <= 1)

        self.config = config
        if "DEBUG_DIADEM" in os.environ:
            logger.error("RUNNING DIADEM IN DEBUG MODE (only 700-710 mz iso windows)")
            scaninfo = scaninfo.filter(pl.col("ms_level") > 1)
            window_filter = [x[0] > 700 and x[0] < 705 for x in scaninfo["iso_window"]]
            scaninfo = scaninfo.filter(window_filter)

        ms1_scaninfo = ms1_scaninfo.drop("collision_energy").with_columns(
            iso_window=0.0,
        )
        self.ms1info = ms1_scaninfo
        self.ms2info = scaninfo.filter(pl.col("ms_level") > 1)
        self.unique_iso_windows = {
            tuple(x) for x in self.ms2info["iso_window"].to_numpy()
        }

    def _preprocess_scangroup(
        self,
        name: str,
        df_chunk: pl.DataFrame,
        mz_range: None | tuple[float, float] = None,
        progress: bool = True,
    ) -> tuple[NDArray, ...]:
        logger.debug(f"Processing iso window {name}")

        window_mzs = []
        window_ints = []
        window_bp_mz = []
        window_bp_int = []
        window_rtinsecs = []
        window_scanids = []

        npeaks_raw = []
        npeaks_deisotope = []

        for row in tqdm(
            df_chunk.rows(named=True),
            desc=f"Preprocessing spectra for {name}",
            disable=not progress,
        ):
            spec_id = row["spec_id"]
            curr_spec: Spectrum = self.adapter[spec_id]
            # NOTE instrument seems to have a wrong value ...
            # Also activation seems to not be recorded ...
            curr_spec = curr_spec.filter_top(self.config.run_max_peaks_per_spec)
            curr_spec = curr_spec.filter_mz_range(*self.config.ion_mz_range)

            mzs = curr_spec.mz
            intensities = curr_spec.intensity

            if mz_range is not None:
                mz_filter = slice(
                    np.searchsorted(mzs, mz_range[0]),
                    np.searchsorted(mzs, mz_range[1], "right"),
                )
                mzs = mzs[mz_filter]
                intensities = intensities[mz_filter]

            # Deisotoping!
            if self.config.run_deconvolute_spectra:
                # Masses need to be ordered for the deisotoping function!
                order = np.argsort(curr_spec.mz)
                npeaks_raw.append(len(order))

                mzs = curr_spec.mz[order]
                intensities = curr_spec.intensity[order]
                if len(mzs) > 0:
                    mzs, intensities = deisotope(
                        mzs,
                        intensities,
                        max_charge=5,
                        diff=self.config.g_tolerances[1],
                        unit=self.config.g_tolerance_units[1],
                    )
                npeaks_deisotope.append(len(mzs))

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

        window_bp_mz = np.array(window_bp_mz).astype(np.float32)
        window_bp_int = np.array(window_bp_int).astype(np.float32)
        window_rtinsecs = np.array(window_rtinsecs).astype(np.float32)
        check_sorted(window_rtinsecs)
        window_scanids = np.array(window_scanids, dtype="object")

        avg_peaks_raw = np.array(npeaks_raw).mean()
        avg_peaks_deisotope = np.array(npeaks_deisotope).mean()
        # Create datasets within each group
        logger.info(f"Saving group {name} with length {len(window_mzs)}")
        logger.info(
            (
                f"{avg_peaks_raw} peaks/spec; {avg_peaks_deisotope} peaks/spec after"
                " deisotoping"
            ),
        )

        out = [
            window_mzs,
            window_ints,
            window_bp_mz,
            window_bp_int,
            window_rtinsecs,
            window_scanids,
        ]
        return out

    def _get_iso_window_group(
        self,
        iso_window_name: str,
        iso_window: tuple[float, float],
        ms2_chunk: pl.DataFrame,
        ms1_chunk: pl.DataFrame,
    ) -> ScanGroup:
        """Gets all spectra that share the same window.

        Gets all spectra that share the same iso window
        and creates a ScanGroup for them.

        Meant for internal usage.

        Parameters
        ----------
        iso_window_name : str
            Name of the isolation window.
        iso_window : tuple[float, float]
            Isolation window.
        ms2_chunk : DataFrame
            DataFrame containing all MS2 spectra.
        ms1_chunk : DataFrame
            DataFrame containing all MS1 spectra.
        """
        (
            window_mzs,
            window_ints,
            window_bp_mz,
            window_bp_int,
            window_rtinsecs,
            window_scanids,
        ) = self._preprocess_scangroup(
            name=iso_window_name,
            df_chunk=ms2_chunk,
            mz_range=None,
        )

        (
            prec_mzs,
            prec_ints,
            _,
            _,
            prec_rtinsecs,
            _,
        ) = self._preprocess_scangroup(
            name="precursors " + iso_window_name,
            df_chunk=ms1_chunk,
            mz_range=iso_window,
        )

        group = ScanGroup(
            iso_window_name=iso_window_name,
            precursor_range=iso_window,
            mzs=window_mzs,
            intensities=window_ints,
            base_peak_mz=window_bp_mz,
            base_peak_int=window_bp_int,
            retention_times=window_rtinsecs,
            scan_ids=window_scanids,
            precursor_intensities=prec_ints,
            precursor_mzs=prec_mzs,
            precursor_retention_times=prec_rtinsecs,
        )
        return group

    def get_iso_window_groups(
        self,
        workerpool: None | Parallel = None,
    ) -> list[ScanGroup]:
        """Returns a list of all ScanGroups in an mzML file."""
        grouped = self.ms2info.sort("RTinSeconds").groupby("iso_window")
        iso_windows, chunks = zip(*list(grouped))
        precursor_info = self.ms1info

        iso_window_names = [
            "({:.06f}, {:.06f})".format(*iso_window) for iso_window in iso_windows
        ]

        if workerpool is None:
            results = [
                self._get_iso_window_group(
                    iso_window_name=iwn,
                    iso_window=iw,
                    ms2_chunk=chunk,
                    ms1_chunk=precursor_info,
                )
                for iwn, iw, chunk in zip(iso_window_names, iso_windows, chunks)
            ]
        else:
            results = workerpool(
                delayed(self._get_iso_window_group)(
                    iso_window_name=iwn,
                    iso_window=iw,
                    ms2_chunk=chunk,
                    ms1_chunk=precursor_info,
                )
                for iwn, iw, chunk in zip(iso_window_names, iso_windows, chunks)
            )

        return results

    def yield_iso_window_groups(self, progress: bool = False) -> Iterator[ScanGroup]:
        """Yield scan groups for each unique isolation window."""
        grouped = self.ms2info.sort("RTinSeconds").groupby("iso_window")
        precursor_info = self.ms1info

        for iso_window, chunk in tqdm(
            grouped,
            disable=not progress,
            desc="Unique Isolation Windows",
        ):
            iso_window_name = "({:.06f}, {:.06f})".format(*iso_window)

            group = self._get_iso_window_group(
                iso_window_name=iso_window_name,
                iso_window=iso_window,
                ms2_chunk=chunk,
                ms1_chunk=precursor_info,
            )
            yield group
