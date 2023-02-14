import numpy as np

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb

diadem_config = DiademConfig()
ms2ml_config = diadem_config.ms2ml_config


def test_peptide_scoring(sample_peaks, albumin_peptides):
    """Tests that the database gets generated correctly from sequences."""
    mzs, ints, z2_mass = sample_peaks

    db = IndexedDb(config=diadem_config, chunksize=64)
    db.targets = albumin_peptides
    db.index_from_sequences()
    scores = db.hyperscore(z2_mass, mzs, ints)
    assert "VPQVSTPTLVEVSR/2" in set(scores["Peptide"])
    return db


def test_database_from_fasta(shared_datadir, sample_peaks):
    """Tests that a database is correctly generated from a fasta file."""
    bsa_fasta = shared_datadir / "BSA.fasta"
    db = IndexedDb(chunksize=64, config=diadem_config)
    db.targets_from_fasta(bsa_fasta)
    db.index_from_sequences()

    mzs, ints, z2_mass = sample_peaks
    scores = db.hyperscore(z2_mass, mzs, ints)
    assert "VPQVSTPTLVEVSR/2" in set(scores["Peptide"][np.invert(scores["decoy"])])
    return db
