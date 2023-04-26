from __future__ import annotations

import time
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from pandas import DataFrame
from tqdm.auto import tqdm

from diadem.config import DiademConfig
from diadem.data_io import read_raw_data
from diadem.data_io.mzml import ScanGroup, StackedChromatograms
from diadem.data_io.timstof import TimsScanGroup, TimsStackedChromatograms
from diadem.index.indexed_db import IndexedDb, db_from_fasta
from diadem.search.search_utils import make_pin
from diadem.utilities.utils import plot_to_log


# @profile
def search_group(  # noqa C901 `search_group` is too complex (18)
    group: ScanGroup | TimsScanGroup,
    db: IndexedDb,
    config: DiademConfig,
    progress: bool = True,
) -> DataFrame:
    """Search a group of scans.

    This function takes a scan group (all scans in a file that share
    isolation window) and does a search using the provided database.

    The search entails iteratvely looking for the highest peak, building
    a stack using it as a center and scoring the traces.
    """
    MAX_PEAKS = config.run_max_peaks  # noqa
    MIN_PEAK_INTENSITY = config.run_min_peak_intensity  # noqa
    DEBUG_FREQUENCY = config.run_debug_log_frequency  # noqa
    ALLOWED_FAILS = config.run_allowed_fails  # noqa
    WINDOWSIZE = config.run_window_size  # noqa
    WINDOW_MAX_PEAKS = config.run_max_peaks_per_window  # noqa

    MIN_INTENSITY_SCALING, MAX_INTENSITY_SCALING = config.run_scalin_limits  # noqa
    SCALING_RATIO = config.run_scaling_ratio  # noqa

    # Min intensity required on a peak in the base
    # scan to be added to the list of mzs that
    # get stacked
    MIN_INTENSITY_RATIO = config.run_min_intensity_ratio  # noqa
    MIN_CORR_SCORE = config.run_min_correlation_score  # noqa

    MS2_TOLERANCE = config.g_tolerances[1]  # noqa
    MS2_TOLERANCE_UNIT = config.g_tolerance_units[1]  # noqa

    MS1_TOLERANCE = config.g_tolerances[0]  # noqa
    MS1_TOLERANCE_UNIT = config.g_tolerance_units[0]  # noqa

    IMS_TOLERANCE = config.g_ims_tolerance  # noqa
    IMS_TOLERANCE_UNIT = config.g_ims_tolerance_unit  # noqa
    MAX_NUM_CONSECUTIVE_FAILS = 50  # noqa

    new_window_kwargs = {
        "window": WINDOWSIZE,
        "min_intensity_ratio": MIN_INTENSITY_RATIO,
        "min_correlation": MIN_CORR_SCORE,
        "mz_tolerance": MS2_TOLERANCE,
        "mz_tolerance_unit": MS2_TOLERANCE_UNIT,
        "max_peaks": WINDOW_MAX_PEAKS,
    }

    if hasattr(group, "imss"):
        logger.info("Detected a diaPASEF dataset")
        new_window_kwargs.update(
            {
                "ims_tolerance": IMS_TOLERANCE,
                "ims_tolerance_unit": IMS_TOLERANCE_UNIT,
            },
        )
        stack_getter = TimsStackedChromatograms.from_group

    else:
        logger.info("No IMS detected")
        stack_getter = StackedChromatograms.from_group

    # Results and stats related variables
    group_results = []
    intensity_log = []
    score_log = []
    index_log = []
    fwhm_log = []
    num_peaks = 0
    num_targets = 0
    num_decoys = 0

    # Fail related variables
    num_fails = 0
    num_consecutive_fails = 0
    curr_highest_peak_int = 2**30
    last_id = None

    pbar = tqdm(desc=f"Slice: {group.iso_window_name}", disable=not progress)
    st = time.time()

    while True:
        if not (curr_highest_peak_int >= MIN_PEAK_INTENSITY and num_peaks <= MAX_PEAKS):
            num_fails += 1

        if num_fails > ALLOWED_FAILS:
            logger.warning(
                (
                    "Exiting scoring loop because number of"
                    f" failes reached the maximum {ALLOWED_FAILS}"
                ),
            )
            break

        if num_consecutive_fails > MAX_NUM_CONSECUTIVE_FAILS:
            logger.warning(
                (
                    "Exiting with early termination due "
                    f"to consecurtive fails {num_consecutive_fails}"
                ),
            )
            group_results = group_results[:-num_consecutive_fails]
            break

        new_stack: StackedChromatograms | TimsStackedChromatograms
        new_stack = group.get_highest_window(**new_window_kwargs)

        if new_stack.base_peak_intensity < MIN_PEAK_INTENSITY:
            break

        if last_id == new_stack.parent_index:
            logger.debug(
                (
                    "Array generated on same index "
                    f"{new_stack.parent_index} as last iteration"
                ),
            )
            num_fails += 1
        else:
            last_id = new_stack.parent_index

        match_id = f"{group.iso_window_name}::{new_stack.parent_index}::{num_peaks}"

        # assert new_stack.base_peak_intensity <= curr_highest_peak_int
        # The current base peak can be higher than the former if several peaks are
        # integrated into the stacked chromatograms.

        curr_highest_peak_int = new_stack.base_peak_intensity
        intensity_log.append(curr_highest_peak_int)
        index_log.append(last_id)
        fwhm_log.append(new_stack.ref_fwhm)

        # scoring_intensities = new_stack.trace_correlation()
        # scoring_intensities = new_stack.center_intensities
        scoring_intensities = (
            new_stack.center_intensities * new_stack.trace_correlation()
        )

        if new_stack.ref_fwhm >= 3:
            scores = db.hyperscore(
                precursor_mz=group.precursor_range,
                spec_int=scoring_intensities,
                spec_mz=new_stack.mzs,
                top_n=1,
                # top_n=100, use 100 when you add precursor information
            )

            # TODO: implement here a
            # partial ms2-score and then a follow up
            # ms1 score

            # if scores is not None:
            #     rt = group.retention_times[new_stack.ref_index]
            #     prec_intensity, prec_dms = group.get_precursor_evidence(
            #         rt,
            #         mzs=scores["PrecursorMZ"].values,
            #         mz_tolerance=MS1_TOLERANCE,
            #         mz_tolerance_unit=MS1_TOLERANCE_UNIT,
            #     )
            #     scores["PrecursorIntensity"] = prec_intensity
            #     scores.drop(
            #         scores[scores.PrecursorIntensity < 100].index,
            #         inplace = True)
            #     scores.reset_index(drop=True, inplace=True)
            #     scores["rank"] = scores["Score"].rank(ascending=False, method="min")
            #     if len(scores) == 0:
            #         logger.debug("All scores were removed due to precursor filtering")
            #         scores = None

        else:
            scores = None

        if scores is not None:
            scores["id"] = match_id
            scores["RetentionTime"] = group.retention_times[new_stack.ref_index]
            if hasattr(group, "imss"):
                scores["IonMobility"] = new_stack.ref_ims
            if scores["decoy"].iloc[0]:
                num_decoys += 1
            else:
                num_targets += 1

            scores = scores.sort_values(by="Score", ascending=False).iloc[:1]
            del scores["rank"]
            match_indices = scores["spec_indices"].iloc[0] + [new_stack.ref_index]
            match_indices = np.sort(np.unique(np.array(match_indices)))

            scaling_window_indices = [
                [x[y] for y in match_indices] for x in new_stack.stack_peak_indices
            ]

            # Scale based on the inverse of the reference chromatogram
            normalized_trace = new_stack.ref_trace / new_stack.ref_trace.max()
            scaling = 1 - normalized_trace
            # Since depending on the sampling rate the peak might be very assymetrical,
            # for now I am not mirroring the scaling
            # scaling = np.stack([scaling, scaling[::-1]], axis=-1).max(axis=-1)
            scaling = np.clip(scaling, MIN_INTENSITY_SCALING, MAX_INTENSITY_SCALING)

            # Scale peaks that match best peptide
            group.scale_window_intensities(
                index=new_stack.parent_index,
                scaling=scaling,
                window_indices=scaling_window_indices,
            )

            if (num_peaks % DEBUG_FREQUENCY) == 0:
                before = new_stack.ref_trace.copy()
                try:
                    s = stack_getter(
                        group=group,
                        index=new_stack.parent_index,
                        **new_window_kwargs,
                    )
                    plot_vals = [before, s.ref_trace]
                    title = (
                        f"Window before ({new_stack.ref_mz}) "
                        f"m/z and after ({s.ref_mz}) m/z"
                    )
                except ValueError:
                    # This happens when all peaks are removed
                    # ie: all in the spectrum matched a peptide
                    # and were removed
                    plot_vals = [before]
                    title = f"Window before ({new_stack.ref_mz}) m/z and after (None)"

                plot_to_log(
                    plot_vals,
                    title=title,
                    lines=True,
                )
                plot_to_log([scaling], title="Scaling")

            group_results.append(scores.copy())
            score_log.append(scores["Score"].max())
            num_consecutive_fails = 0
        else:
            logger.debug(f"{match_id} did not match any peptides, scaling and skipping")
            scaling = (
                SCALING_RATIO
                * np.ones_like(new_stack.ref_trace)
                * MIN_INTENSITY_SCALING
            )
            group.scale_window_intensities(
                index=new_stack.parent_index,
                scaling=scaling,
                window_indices=new_stack.stack_peak_indices,
            )
            num_fails += 1
            num_consecutive_fails += 1

        num_peaks += 1
        pbar.set_postfix(
            {
                "last_id": last_id,
                "max_intensity": curr_highest_peak_int,
                "num_fails": num_fails,
                "num_scores": len(group_results),
                "n_targets": num_targets,
                "n_decoys": num_decoys,
            },
        )
        pbar.update(1)

        # TODO move this so it is disabled without the debug flag ...
        if ((et := time.time()) - st) >= 2:
            from ms2ml.utils.mz_utils import get_tolerance

            tot_candidates = 0
            for mz in new_stack.mzs:
                ms2_tol = get_tolerance(
                    MS2_TOLERANCE,
                    theoretical=mz,
                    unit=MS2_TOLERANCE_UNIT,
                )

                candidates = db.bucketlist.yield_candidates(
                    ms1_range=group.precursor_range,
                    ms2_range=(mz - ms2_tol, mz + ms2_tol),
                )
                for _ in candidates:
                    tot_candidates += 1

            logger.error(
                (
                    f"Iteration took waaaay too long scores={scores} ;"
                    f" {tot_candidates} total candidates for precursor range "
                    f"{group.precursor_range} and m/z range "
                    f"and ms2 range {(mz - ms2_tol, mz + ms2_tol)}"
                ),
            )
            logger.error(f"{new_stack.mzs.copy()}; len({len(new_stack.mzs)})")
            st = et
        else:
            st = et

    pbar.close()
    plot_to_log(
        np.log1p(np.array(intensity_log)),
        title="Max (log) intensity over time",
    )
    plot_to_log(np.array(score_log), title="Score over time")
    plot_to_log(np.array(index_log), title="Requested index over time")
    plot_to_log(np.array(fwhm_log), title="FWHM across time")
    logger.info(
        (
            f"Done with window {group.iso_window_name}, "
            f"scored {num_peaks} peaks in {len(group.base_peak_int)} spectra. "
            f"Intensity of the last scored peak {curr_highest_peak_int} "
            f"on index {last_id}"
        ),
    )

    if len(group_results) == 0:
        logger.error("No results were accumulated in this group!")
    group_results = pd.concat(group_results) if len(group_results) else None
    return group_results


# @profile
def diadem_main(
    fasta_path: Path | str,
    data_path: Path | str,
    config: DiademConfig,
    out_prefix: str = "",
) -> None:
    """Main function for running Diadem.

    Parameters
    ----------
    fasta_path : Path | str
        Path to the fasta file
    data_path : Path | str
        Path to the mzml file or .d directory.
    config : DiademConfig
        Configuration object to use for the run.
    out_prefix : str, optional
        Prefix to use for the output files, by default ""
        For example if nothing is passed, the ourpur will be diadem.csv,
        othwerise, if out_prefix is "test", the output will be test.diadem.csv
    """
    start_time = time.time()

    # Set up database
    db, cache = db_from_fasta(
        fasta=fasta_path,
        chunksize=None,
        config=config,
        index=False,
    )

    # set up mzml file
    ss = read_raw_data(
        filepath=data_path,
        config=config,
    )

    results = []

    if config.run_parallelism == 1:
        for group in ss.yield_iso_window_groups(progress=True):
            group_db = db.index_prefiltered_from_parquet(cache, *group.precursor_range)
            group_results = search_group(group=group, db=group_db, config=config)
            results.append(group_results)
    else:

        @delayed
        def setup_db_and_search(
            precursor_range: tuple[float, float],
            db: IndexedDb,
            cache_location: PathLike,
            config: DiademConfig,
            group: ScanGroup,
        ) -> DataFrame:
            pfdb = db.index_prefiltered_from_parquet(cache_location, *precursor_range)
            results = search_group(group=group, db=pfdb, config=config)
            return results

        with Parallel(n_jobs=config.run_parallelism) as workerpool:
            groups = ss.get_iso_window_groups(workerpool=workerpool)
            precursor_ranges = [group.precursor_range for group in groups]
            results = workerpool(
                setup_db_and_search(
                    precursor_range=prange,
                    db=db,
                    cache_location=cache,
                    config=config,
                    group=group,
                )
                for group, prange in zip(groups, precursor_ranges)
            )

    results: pd.DataFrame = pd.concat(
        [x for x in results if x is not None],
        ignore_index=True,
    )

    prefix = out_prefix + ".diadem" if out_prefix else "diadem"
    Path(prefix).absolute().parent.mkdir(exist_ok=True)

    logger.info(f"Writting {prefix+'.csv'} and {prefix+'.parquet'}")
    results.to_csv(prefix + ".csv", index=False)
    results.to_parquet(prefix + ".parquet", index=False, engine="pyarrow")
    make_pin(
        results,
        fasta_path=fasta_path,
        mzml_path=data_path,
        pin_path=prefix + ".tsv.pin",
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
