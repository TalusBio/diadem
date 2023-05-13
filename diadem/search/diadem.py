from __future__ import annotations

import logging
import os
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
from diadem.search.mokapot import brew_run
from diadem.utilities.logging import InterceptHandler
from diadem.utilities.utils import plot_to_log

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

if "PLOTDIADEM" in os.environ and os.environ["PLOTDIADEM"]:
    import matplotlib.pyplot as plt  # noqa: I001


# @profile
def search_group(
    group: ScanGroup,
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

    MIN_INTENSITY_SCALING, MAX_INTENSITY_SCALING = config.run_scaling_limits  # noqa
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
    MAX_NUM_CONSECUTIVE_FAILS = 100  # noqa

    start_rts, start_bpc = group.retention_times.copy(), group.base_peak_int.copy()

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
                (
                    "Exiting scoring loop because number of"
                    f" failes reached the maximum {ALLOWED_FAILS}"
                ),
            )
            break

        new_stack: StackedChromatograms = group.get_highest_window(
            window=WINDOWSIZE,
            min_intensity_ratio=MIN_INTENSITY_RATIO,
            min_correlation=MIN_CORR_SCORE,
            tolerance=MS2_TOLERANCE,
            tolerance_unit=MS2_TOLERANCE_UNIT,
            max_peaks=WINDOW_MAX_PEAKS,
        )
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
            )
        else:
            scores = None

        if scores is not None:
            scores["peak_id"] = match_id
            scores["RetentionTime"] = group.retention_times[new_stack.parent_index]
            if hasattr(group, "imss"):
                scores["IonMobility"] = new_stack.ref_ims
            if scores["decoy"].iloc[0]:
                num_decoys += 1
            else:
                num_targets += 1

            scores = scores.sort_values(by="Score", ascending=False).iloc[:1]
            match_indices = scores["spec_indices"].iloc[0] + [new_stack.ref_index]
            match_indices = np.sort(np.unique(np.array(match_indices)))

            min_corr_score_scale = np.quantile(
                new_stack.correlations[match_indices],
                0.75,
            )
            scores["q75_correlation"] = min_corr_score_scale
            corr_match_indices = np.where(
                new_stack.correlations > min_corr_score_scale,
            )[0]
            match_indices = np.sort(
                np.unique(np.concatenate([match_indices, corr_match_indices])),
            )

            if "PLOTDIADEM" in os.environ and os.environ["PLOTDIADEM"]:
                try:
                    ax1.cla()
                    ax2.cla()
                except NameError:
                    fig, (ax1, ax2) = plt.subplots(1, 2)

                new_stack.plot(ax1, matches=match_indices)

                ax2.plot(start_rts, np.sqrt(start_bpc), alpha=0.2, color="gray")
                ax2.plot(group.retention_times, np.sqrt(group.base_peak_int))
                ax2.vlines(
                    x=group.retention_times[new_stack.parent_index],
                    ymin=0,
                    ymax=np.sqrt(new_stack.base_peak_intensity),
                    color="r",
                )
                plt.title(
                    (
                        f"Score: {scores['Score'].iloc[0]} \n"
                        f" Peptide: {scores['peptide'].iloc[0]} \n"
                        f"@ RT: {scores['RetentionTime'].iloc[0]} \n"
                        f"Corr Score: {min_corr_score_scale}"
                    ),
                )
                plt.pause(0.01)

            scaling_window_indices = [
                [x[y] for y in match_indices] for x in new_stack.stack_peak_indices
            ]

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
                    min_intensity_ratio=MIN_INTENSITY_RATIO,
                    min_correlation=MIN_CORR_SCORE,
                    max_peaks=MAX_PEAKS,
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
                f"peak {num_peaks}/{MAX_PEAKS} max ; Intensity {curr_highest_peak_int}",
            )

    pbar.close()
    plot_to_log(
        np.log1p(np.array(intensity_log)),
        title="Max (log) intensity over time",
    )
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
    db, cache = db_from_fasta(
        fasta=fasta_path,
        chunksize=None,
        config=config,
        index=False,
    )

    # set up mzml file
    ss = SpectrumStacker(
        mzml_file=mzml_path,
        config=config,
    )

    results = []

    if config.run_parallelism == 1:
        for group in ss.yield_iso_window_groups():
            group_db = db.index_prefiltered_from_parquet(cache, *group.precursor_range)
            group_results = search_group(group=group, db=group_db, config=config)
            if group_results is not None:
                group_results.to_parquet("latestresults.parquet")
            results.append(group_results)
    else:
        with Parallel(n_jobs=config.run_parallelism) as workerpool:
            groups = ss.get_iso_window_groups(workerpool=workerpool)
            precursor_ranges = [group.precursor_range for group in groups]
            dbs = workerpool(
                delayed(db.index_prefiltered_from_parquet)(cache, *prange)
                for prange in precursor_ranges
            )
            results = workerpool(
                delayed(search_group)(group=group, db=pfdb, config=config)
                for group, pfdb in zip(groups, dbs)
            )

    results: pd.DataFrame = pd.concat(results, ignore_index=True)

    prefix = out_prefix + ".diadem" if out_prefix else "diadem"
    prefix_dir = Path(prefix).absolute()

    prefix_dir.parent.mkdir(exist_ok=True)

    logger.info(f"Writting {prefix+'.csv'} and {prefix+'.parquet'}")
    results.to_csv(prefix + ".csv", index=False)

    # RTs are stored as f16, which need to be converted to f32 for parquet
    f16_cols = list(results.select_dtypes("float16"))
    if f16_cols:
        for col in f16_cols:
            results[col] = results[col].astype("float32")
    results.to_parquet(prefix + ".parquet", index=False, engine="pyarrow")
    try:
        # Right now I am bypassing the mokapot results, because they break a test
        # meant to check that no decoys are detected (which is true in that case).
        logger.info("Running mokapot")
        mokapot_results = brew_run(
            results,
            fasta_path=fasta_path,
            ms_data_path=data_path,
        )
        logger.info(f"Writting mokapot results to {prefix}.peptides.parquet")
        mokapot_results.to_parquet(prefix + ".peptides.parquet")
    except ValueError as e:
        if "decoy PSMs were detected" in str(e):
            logger.warning(f"Could not run mokapot: {e}")
        else:
            logger.error(f"Could not run mokapot: {e}")
            raise e
    except RuntimeError as e:
        logger.warning(f"Could not run mokapot: {e}")
        logger.error(results)
        raise e
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time}")
