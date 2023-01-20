import sqlite3

import pandas as pd

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb

diadem_config = DiademConfig()
ms2ml_config = diadem_config.ms2ml_config


# Sample peaks is defined in conftest.py
def test_sql_generation(shared_datadir, tmpdir, sample_peaks):
    """Tests that generating a database from a cached sql works."""
    bsa_fasta = shared_datadir / "BSA.fasta"
    sql_path = tmpdir / "cache.sqlite"
    db = IndexedDb(chunksize=64, config=diadem_config)
    db.targets_from_fasta(bsa_fasta)
    db.gen_to_sqlite(sqlite_db=sql_path)

    conn = sqlite3.Connection(sql_path)
    df = pd.read_sql_query(
        "SELECT * FROM fragments", conn, dtype={"mz": "float32", "seq_id": "int"}
    )
    assert len(df) > 2000
    assert df["mz"].array.dtype == "float32"
    conn.close()

    db.index_from_sqlite(sqlite_path=sql_path)
    mzs, ints, z2_mass = sample_peaks
    scores = db.hyperscore(z2_mass, mzs, ints)
    assert "VPQVSTPTLVEVSR/2" in set(scores["Peptide"])
