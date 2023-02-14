import numpy as np
from ms2ml.data.adapters import FastaAdapter

from diadem.config import DiademConfig
from diadem.index.indexed_db import IndexedDb


def test_fasta_shows_in_db(shared_datadir):
    """Tests matching theoretical peptides to a database.

    Tests that when creating a database from BSA,
    all peptides are matched to a theoretical spectrum
    with ones for the intensity of all ions.
    """
    config = DiademConfig()
    ms2ml_config = config.ms2ml_config
    adapter = FastaAdapter(
        shared_datadir / "BSA.fasta", config=ms2ml_config, only_unique=True
    )
    sequences = list(adapter.parse())

    db = IndexedDb(config=config, chunksize=512)
    db.targets = sequences
    db.index_from_sequences()

    for i, s in enumerate(sequences):
        mzs = s.theoretical_ion_masses
        intens = np.ones_like(mzs, dtype="float")

        ms1_range = (s.mz - 10, s.mz + 10)
        score_df = db.hyperscore(ms1_range, spec_mz=mzs, spec_int=intens, top_n=2)
        score_df = score_df[np.invert(score_df["decoy"])]
        assert s.to_proforma() in list(
            score_df["Peptide"]
        ), f"Peptide i={i} {s.to_proforma()} not in db"
