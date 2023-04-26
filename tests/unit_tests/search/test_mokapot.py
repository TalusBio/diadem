"""Unit tests for mokapot interactions."""
import pandas as pd
import pytest

from diadem.index.protein_index import ProteinNGram
from diadem.search.mokapot import _decoy_to_target, _get_proteins, _prepare_df


@pytest.fixture
def kid_fasta(tmp_path):
    """A tiny fasta."""
    fasta = """
    > sp|KID1|KID1_HUMAN
    LESLIEKAAAAAR
    > sp|KID3|KID3_HUMAN
    EDITHKAAAAAR
    """

    fasta_file = tmp_path / "test.fasta"
    with fasta_file.open("w+") as fout:
        fout.write(fasta)

    return fasta_file


def test_prepare_df(kid_fasta):
    """Test that our dataframe is prepared correctly."""
    in_df = pd.DataFrame(
        {
            "rank": [1, 2, 1, 1, 1],
            "peptide": ["EDITH", "EDITH", "LES[+79.9]LIE", "LILS[+79.9]EE", "AAAR"],
            "list_col": [[1, 2]] * 5,
            "cool_npeaks": [5, 5, 6, 6, 4],
            "decoy": [False, False, False, True, False],
        },
    )

    expected = pd.DataFrame(
        {
            "peptide": ["EDITH", "LES[+79.9]LIE", "LILS[+79.9]EE", "AAAR"],
            "cool_npeaks": [5, 6, 6, 4],
            "is_target": [True, True, False, True],
            "filename": "test",
            "target_pair": ["EDITH", "LES[+79.9]LIE", "LES[+79.9]LIE", "AAAR"],
            "peptide_length": [5, 6, 6, 4],
            "cool_npeaks_pct": [100.0, 100.0, 100.0, 100.0],
            "proteins": ["KID3", "KID1", "KID1", "KID1;KID3"],
        },
        index=[0, 2, 3, 4],
    )

    out_df = _prepare_df(in_df, kid_fasta, "test.mzML")
    pd.testing.assert_frame_equal(out_df, expected)


def test_decoy_to_target():
    """Test that our decoy to target function works correctly."""
    target = "LES[+79.9]LIEK"

    # Test reversal
    decoy = "LEILS[+79.9]EK"
    assert target == _decoy_to_target(decoy)

    # Test another permutation:
    perm = [2, 0, 1, 3, 4, 5, 6]
    decoy = "S[+79.9]LELIEK"

    assert target == _decoy_to_target(decoy, perm)


def test_get_proteins(kid_fasta):
    """Test that _get_proteins works corectly."""
    ngram = ProteinNGram.from_fasta(kid_fasta)
    assert _get_proteins("LESLIE", ngram) == "KID1"
    assert _get_proteins("EDITH", ngram) == "KID3"
    assert _get_proteins("AAAR", ngram) == "KID1;KID3"
