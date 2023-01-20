from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger
from ms2ml import Spectrum
from ms2ml.data.adapters import MZMLAdapter
from pandas import DataFrame
from tqdm.auto import tqdm

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb, db_from_fasta


def score(db: IndexedDb, spec: Spectrum, mzml_stem: str) -> DataFrame | None:
    """Score a spectrum against a database.

    Utility function that adds some extra infromation as context to the score
    in the dda search.
    """
    if spec is None:
        return None
    spec_results = db.hyperscore(
        spec.precursor_mz, spec_mz=spec.mz, spec_int=spec.intensity, top_n=10
    )
    if spec_results is not None:
        spec_results["ScanID"] = f"{mzml_stem}::{spec.extras['id']}"
        return spec_results


def main(
    mzml_path: Path | str,
    fasta_path: Path | str,
    config: DiademConfig,
    out_prefix: str = "",
) -> None:
    """Run the DDA mode of DIAdem."""
    # TODO make this a parameter ... maybe DDA mz range/DIA mz range?
    # Maybe just DDA and use the the range from the mzML for DIA?
    replace(config, peptide_mz_range=(350.0, 2000.0))

    start_time = time.time()

    # Set up database
    db = db_from_fasta(
        fasta=fasta_path,
        config=config,
        chunksize=config.db_bucket_size,
    )
    db.bucketlist.sort("ms1")

    def out_hook(spec: Spectrum) -> Spectrum | None:
        if spec.ms_level != 2:
            return None
        elif len(spec.mz) < 15:
            return None
        else:
            return spec.filter_top(150)

    # set up mzml file
    mzml = MZMLAdapter(
        file=mzml_path,
        config=config,
        out_hook=out_hook,
    )
    mzml_stem = Path(mzml_path).stem

    results = []

    for spec in tqdm(mzml.parse(), desc="Scoring Spectra"):
        if spec is None:
            continue
        spec_results = db.hyperscore(
            spec.precursor_mz, spec_mz=spec.mz, spec_int=spec.intensity, top_n=10
        )
        if spec_results is not None:
            spec_results["ScanID"] = f"{mzml_stem}::{spec.extras['id']}"
            results.append(spec_results)

    results = pd.concat([r for r in results if r is not None])
    prefix = out_prefix + ".diadem.dda" if out_prefix else "diadem.dda"
    logger.info(f"Writting {prefix+'.csv'} and {prefix+'.parquet'}")
    results.to_csv(prefix + ".csv", index=False)
    results.to_parquet(prefix + ".parquet", index=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
