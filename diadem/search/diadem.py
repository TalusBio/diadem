from __future__ import annotations

import itertools
import time
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import uniplot
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


def plot_to_log(*args, **kwargs) -> None:  # noqa
    """Plot to log.

    Generates a plot of the passed data to the function.
    All arguments are passed internally to uniplot.plot_to_string.
    """
    for line in uniplot.plot_to_string(*args, **kwargs):
        logger.debug(line)


# @profile
def search_group(
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

    IMS_TOLERANCE = config.g_ims_tolerance  # noqa
    IMS_TOLERANCE_UNIT = config.g_ims_tolerance_unit  # noqa

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
            }
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

    # Fail related variables
    num_fails = 0
    curr_highest_peak_int = 2**30
    last_id = None

    pbar = tqdm(desc=f"Slice: {group.iso_window_name}", disable=not progress)

    while True:
        if not (curr_highest_peak_int >= MIN_PEAK_INTENSITY and num_peaks <= MAX_PEAKS):
            num_fails += 1

        if num_fails > ALLOWED_FAILS:
            logger.warning(
                "Exiting scoring loop because number of"
                f" failes reached the maximum {ALLOWED_FAILS}"
            )
            break

        new_stack: StackedChromatograms | TimsStackedChromatograms
        new_stack = group.get_highest_window(**new_window_kwargs)
        if new_stack.base_peak_intensity < MIN_PEAK_INTENSITY:
            break

        if last_id == new_stack.parent_index:
            logger.debug(
                "Array generated on same index "
                f"{new_stack.parent_index} as last iteration"
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
            )
        else:
            scores = None

        if scores is not None:
            score_log.append(scores["Score"].max())
            scores["id"] = match_id
            ref_peak_mz = new_stack.mzs[new_stack.ref_index]

            mzs = itertools.chain(
                *[scores[x].iloc[0] for x in scores.columns if "_mzs" in x]
            )
            best_match_mzs = np.sort(
                np.array(tuple(itertools.chain(mzs, [ref_peak_mz])))
            )

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
                mzs=best_match_mzs,
                window_indices=new_stack.stack_peak_indices,
                window_mzs=new_stack.mzs,
            )

            if (num_peaks % DEBUG_FREQUENCY) == 0:
                before = new_stack.ref_trace.copy()
                s = stack_getter(
                    group=group,
                    index=new_stack.parent_index,
                    **new_window_kwargs,
                )
                plot_to_log(
                    [before, s.ref_trace],
                    title=(
                        f"Window before ({new_stack.ref_mz}) "
                        f"m/z and after ({s.ref_mz}) m/z"
                    ),
                    lines=True,
                )
                plot_to_log([scaling], title="Scaling")

            group_results.append(scores.copy())
        else:
            logger.debug(f"{match_id} did not match any peptides, scaling and skipping")
            scaling = SCALING_RATIO * np.ones_like(new_stack.ref_trace)
            group.scale_window_intensities(
                index=new_stack.parent_index,
                scaling=scaling,
                mzs=new_stack.mzs,
                window_indices=new_stack.stack_peak_indices,
                window_mzs=new_stack.mzs,
            )
            num_fails += 1

        num_peaks += 1
        pbar.set_postfix(
            {
                "last_id": last_id,
                "max_intensity": curr_highest_peak_int,
                "num_fails": num_fails,
                "num_scores": len(group_results),
            }
        )
        pbar.update(1)
        if (num_peaks % DEBUG_FREQUENCY) == 0:
            logger.debug(
                f"peak {num_peaks}/{MAX_PEAKS} max ; Intensity {curr_highest_peak_int}"
            )

    pbar.close()
    plot_to_log(
        np.log1p(np.array(intensity_log)), title="Max (log) intensity over time"
    )
    plot_to_log(np.array(score_log), title="Score over time")
    plot_to_log(np.array(index_log), title="Requested index over time")
    plot_to_log(np.array(fwhm_log), title="FWHM across time")
    logger.info(
        f"Done with window {group.iso_window_name}, "
        f"scored {num_peaks} peaks in {len(group.base_peak_int)} spectra. "
        f"Intensity of the last scored peak {curr_highest_peak_int} "
        f"on index {last_id}"
    )
    group_results = pd.concat(group_results)
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
        fasta=fasta_path, chunksize=None, config=config, index=False
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

    results: pd.DataFrame = pd.concat(results, ignore_index=True)

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
