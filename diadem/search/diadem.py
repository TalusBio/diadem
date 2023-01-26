from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uniplot
from joblib import Parallel, delayed
from loguru import logger
from pandas import DataFrame
from tqdm.auto import tqdm

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb, db_from_fasta
from diadem.mzml import ScanGroup, SpectrumStacker, StackedChromatograms


def plot_to_log(*args, **kwargs) -> None:  # noqa
    """Plot to log.

    Generates a plot of the passed data to the function.
    All arguments are passed internally to uniplot.plot_to_string.
    """
    for line in uniplot.plot_to_string(*args, **kwargs):
        logger.debug(line)


# @profile
def search_group(
    group: ScanGroup, db: IndexedDb, config: DiademConfig, progress: bool = True
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

    MIN_INTENSITY_SCALING, MAX_INTENSITY_SCALING = config.run_scalin_limits  # noqa
    SCALING_RATIO = config.run_scaling_ratio  # noqa

    # Min intensity required on a peak in the base
    # scan to be added to the list of mzs that
    # get stacked
    MIN_INTENSITY_RATIO = config.run_min_intensity_ratio  # noqa
    MIN_CORR_SCORE = config.run_min_correlation_score  # noqa

    MS2_TOLERANCE = config.g_tolerances[1]  # noqa
    MS2_TOLERANCE_UNIT = config.g_tolerance_units[1]  # noqa

    # Results and stats related variables
    group_results = []
    intensity_log = []
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

        new_stack: StackedChromatograms = group.get_highest_window(
            window=WINDOWSIZE,
            min_intensity_ratio=MIN_INTENSITY_RATIO,
            min_correlation=MIN_CORR_SCORE,
            tolerance=MS2_TOLERANCE,
            tolerance_unit=MS2_TOLERANCE_UNIT,
        )
        if new_stack.base_peak_intensity < MIN_PEAK_INTENSITY:
            break

        if last_id is new_stack.parent_index:
            logger.debug(
                "Array generated on same index "
                f"{new_stack.parent_index} as last iteration"
            )
            num_fails += 1
        else:
            last_id = new_stack.parent_index

        match_id = f"{group.iso_window_name}::{new_stack.parent_index}::{num_peaks}"
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
                top_n=10,
            )
        else:
            scores = None

        if scores is not None:
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
            # scaling = SCALING_RATIO * (1-normalized_trace)
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
                s = StackedChromatograms.from_group(
                    group=group,
                    index=new_stack.parent_index,
                    window=WINDOWSIZE,
                    tolerance=db.config.g_tolerances[1],
                    tolerance_unit=db.config.g_tolerance_units[1],
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
            logger.info(f"{match_id} did not match any peptides, scaling and skipping")
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
        pbar.update(1)
        if (num_peaks % DEBUG_FREQUENCY) == 0:
            logger.debug(
                f"peak {num_peaks}/{MAX_PEAKS} max ; Intensity {curr_highest_peak_int}"
            )

    pbar.close()
    plot_to_log(
        np.log1p(np.array(intensity_log)), title="Max (log) intensity over time"
    )
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
    mzml_path: Path | str,
    config: DiademConfig,
    out_prefix: str = "",
) -> None:
    """Main function for running Diadem.

    Parameters
    ----------
    fasta_path : Path | str
        Path to the fasta file
    mzml_path : Path | str
        Path to the mzml file
    config : DiademConfig
        Configuration object to use for the run.
    out_prefix : str, optional
        Prefix to use for the output files, by default ""
        For example if nothing is passed, the ourpur will be diadem.csv,
        othwerise, if out_prefix is "test", the output will be test.diadem.csv
    """
    start_time = time.time()

    # Set up database
    db = db_from_fasta(
        fasta=fasta_path,
        config=config,
        # chunksize=2**11,
        chunksize=2**15,
    )

    # set up mzml file
    ss = SpectrumStacker(
        mzml_file=mzml_path,
        config=config,
    )

    results = []

    # This is a very easy point of parallelism
    if config.run_parallelism == 1:
        for group in ss.yield_iso_window_groups(progress=True):
            group_db = db.prefilter_ms1(group.precursor_range)
            group_results = search_group(group=group, db=group_db, config=config)
            results.append(group_results)
    else:
        # with Parallel(n_jobs=config.run_parallelism) as workerpool:
        with Parallel(n_jobs=4) as workerpool:
            groups = ss.get_iso_window_groups(workerpool=workerpool)
            dbs = [db.prefilter_ms1(group.precursor_range) for group in groups]
            results = workerpool(
                delayed(search_group)(group=group, db=pfdb, config=config)
                for group, pfdb in zip(groups, dbs)
            )

    results = pd.concat(results, ignore_index=True)
    prefix = out_prefix + ".diadem" if out_prefix else "diadem"
    logger.info(f"Writting {prefix+'.csv'} and {prefix+'.parquet'}")
    results.to_csv(prefix + ".csv", index=False)
    results.to_parquet(prefix + ".parquet", index=False, engine="pyarrow")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
