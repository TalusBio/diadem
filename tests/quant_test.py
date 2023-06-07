import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import diadem.quantify.quant as q


def test_quant(get_data):
    """Test that the entire module returns the correct data."""
    ms_data, pep = get_data
    quant_results = q.quant(ms_data, pep)
    manual_results = pl.DataFrame(
        {
            "peptide": ["PEPTIDEEEE", "LESLIE", "EDITH"],
            "intensity": [
                np.trapz([610.00, 100.00]),
                np.trapz([1111.00, (2210.00 + 1560.00), 4000.00]),
                np.trapz(
                    [
                        110.8,
                        110.8,
                        110.8,
                        123.1,
                        456.4,
                        765.7,
                        321.3,
                        110.8,
                        110.8,
                        110.8,
                    ]
                ),
            ],
            "mz": [100.01, np.average([249.99, 249.985]), 100.0],
            "num_fragments": [2.0, 3.0, 10.0],
        }
    )
    assert_frame_equal(manual_results, quant_results)


def test_match_peptide(get_data):
    """Test that the module correctly matches the peptides to the spectra."""
    ms_data, peptides = get_data
    match_result = q.match_peptide(ms_data, peptides)
    manual_result = pl.DataFrame(
        {
            "peptide": ["PEPTIDEEEE", "LESLIE", "EDITH"],
            "rt": [100.0, 150.0, 5.0],
            "intensity": [610.00, (2210.00 + 1560.00), 123.1],
            "mz": [100.01, np.average([249.99, 249.985]), 100.0],
            "ims": [99.982, np.average([149.98, 150.01]), 100.0],
        }
    )
    assert_frame_equal(match_result, manual_result)


def test_get_peaks(get_data):
    """Test that the quant module is accurately identifying the
    corresponding peaks at different rts.
    """
    ms_data, peptides = get_data
    pep_spectrum_df = q.match_peptide(ms_data, peptides)
    row = pep_spectrum_df.row(
        by_predicate=(pl.col("peptide") == "PEPTIDEEEE"), named=True
    )
    num_ms_data = ms_data.with_row_count()

    r_intensities, r_rts = q.get_peaks(num_ms_data, row, 1, 100.0)
    r_intensities_manual = [100.0]
    r_rts_manual = [110.0]
    assert r_intensities == r_intensities_manual and r_rts == r_rts_manual

    l_intensities, l_rts = q.get_peaks(num_ms_data, row, 1, 0, False)
    l_intensities_manual = []
    l_rts_manual = []
    assert l_intensities == l_intensities_manual and l_rts == l_rts_manual
