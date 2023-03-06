import numpy as np
import pandas as pd

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb

diadem_config = DiademConfig()
ms2ml_config = diadem_config.ms2ml_config


# Sample peaks is defined in conftest.py
def test_parquet_generation(shared_datadir, tmpdir, sample_peaks):
    """Tests that generating a database from a cached sql works."""
    bsa_fasta = shared_datadir / "BSA.fasta"
    db = IndexedDb(chunksize=64, config=diadem_config)
    db.targets_from_fasta(bsa_fasta)
    db.generate_to_parquet(tmpdir)

    df = pd.read_parquet(tmpdir / "frags.parquet")
    assert len(df) > 2000
    assert df["mz"].array.dtype == "float64"

    df = pd.read_parquet(tmpdir / "seqs.parquet")
    assert len(df) > 20
    assert len(df) < 2000

    db.index_from_parquet(tmpdir)
    mzs, ints, z2_mass = sample_peaks
    scores = db.hyperscore(z2_mass, mzs, ints)
    assert "VPQVSTPTLVEVSR/2" in set(scores["Peptide"])

    pep_scores = scores[scores["Peptide"] == "VPQVSTPTLVEVSR/2"]
    assert all(np.invert(pep_scores["decoy"]))
