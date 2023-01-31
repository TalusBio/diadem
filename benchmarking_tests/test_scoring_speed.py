import pytest
from tqdm.auto import tqdm

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb, db_from_fasta


@pytest.fixture(params=[2**9, 2**11, 2**13])
def fake_database(fake_5k_fasta, request):
    db = db_from_fasta(
        fake_5k_fasta, chunksize=request.param, config=DiademConfig(run_parallelism=1)
    )
    return db


@pytest.fixture(params=[2, 3, 4])
def fake_prefiltered_database(fake_database: IndexedDb, request):
    db = fake_database.prefilter_ms1((700, 720), num_decimals=request.param)
    return db


def score_all_specs_open(db: IndexedDb, specs):
    for mzs, ints, prec_mz in tqdm(specs):
        db.hyperscore(precursor_mz=(700.0, 720.0), spec_int=ints, spec_mz=mzs)


def test_db_scoring_speed_unfiltered(fake_database, fake_spectra_tuples_100, benchmark):
    """Benchmarks how long it takes to search 100 spectra on a database over a range of 20da.
    """
    benchmark(score_all_specs_open, fake_database, fake_spectra_tuples_100)


def test_db_scoring_speed_filtered(
    fake_prefiltered_database, fake_spectra_tuples_100, benchmark
):
    """Benchmarks how long it takes to search 100 spectra on a prefiltered database over a range of 20da.
    """
    benchmark(score_all_specs_open, fake_prefiltered_database, fake_spectra_tuples_100)


def score_all_specs_closed(db: IndexedDb, specs):
    for mzs, ints, prec_mz in tqdm(specs):
        db.hyperscore(
            precursor_mz=(prec_mz - 0.01, prec_mz + 0.01), spec_int=ints, spec_mz=mzs
        )


def test_db_scoring_speed_closed(fake_database, fake_spectra_tuples_100, benchmark):
    """Benchmarks how long it would take to search spectra in closed searches (0.01da on either side).
    """
    benchmark(score_all_specs_closed, fake_database, fake_spectra_tuples_100)
