import numpy as np
import pandas as pd
import pytest
from ms2ml import Peptide

from diadem.config import DiademConfig
from diadem.search.diadem import diadem_main


@pytest.mark.parametrize("parallel", [False, True], ids=["NoParallel", "Parallel"])
def test_dia_search_works_mzml(tmpdir, shared_datadir, parallel):
    """Uses simulated data to test diadem.

    Uses data simulated using Synthedia to check if the full search
    can recall all peptides from FGFR1 in the simulated dataset.
    """
    fasta = shared_datadir / "mzml/P11362.fasta"
    mzml = shared_datadir / "mzml/FGFR1_600_800_5min_group_0_sample_0.mzML"
    config = DiademConfig(run_parallelism=2 if parallel else 1)
    out = str(tmpdir / "out")
    diadem_main(config=config, data_path=mzml, fasta_path=fasta, out_prefix=out)

    expected_csv = out + ".diadem.csv"
    df = pd.read_csv(expected_csv)
    df = df[np.invert(df["decoy"])]
    peptides = {Peptide.from_sequence(x).stripped_sequence for x in df.Peptide.unique()}

    theo_table = pd.read_csv(
        shared_datadir / "mzml/FGFR1_600_800_5min_peptide_table.tsv", sep="\t"
    )
    theo_table = theo_table[
        theo_table["MS2 chromatographic points group_0_sample_0"] > 0
    ]

    theo_table["seen_seqs"] = [x in peptides for x in theo_table.Sequence]
    missed_theo = theo_table[[not x for x in theo_table["seen_seqs"]]]
    seen_theo = theo_table[list(theo_table["seen_seqs"])]

    assert len(missed_theo) == 0
    assert len(seen_theo) > 1
