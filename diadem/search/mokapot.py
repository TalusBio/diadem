"""Run-level mokapot analyses."""
import re
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import mokapot
import pandas as pd

from diadem.index.protein_index import ProteinNGram


def brew_run(
    results: pd.DataFrame,
    fasta_path: PathLike,
    ms_data_path: PathLike,
) -> pd.DataFrame:
    """Prepare the result DataFrame for mokapot.

    Parameters
    ----------
    results: pd.DataFrame
        The diadem search results.
    fasta_path : PathLike
        The FASTA file that was used for the search.
    ms_data_path : PathLike
        The mass spectrometry data file that was searched.

    Returns
    -------
    pd.DataFrame
        The run-level peptide scores and confidence estimates.
    """
    input_df = _prepare_df(results, fasta_path, ms_data_path)
    nonfeat = [
        "peptide",
        "proteins",
        "filename",
        "target_pair",
        "peak_id",
        "is_target",
    ]
    peptides = mokapot.LinearPsmDataset(
        psms=input_df,
        target_column="is_target",
        spectrum_columns="target_pair",
        peptide_column="peptide",
        protein_column="proteins",
        feature_columns=[c for c in input_df.columns if c not in nonfeat],
        filename_column="filename",
        copy_data=False,
    )
    results = mokapot.brew(peptides)
    return results.peptides


def _prepare_df(
    results: pd.DataFrame,
    fasta_path: PathLike,
    ms_data_path: PathLike,
) -> pd.DataFrame:
    """Prepare the result DataFrame for mokapot.

    Parameters
    ----------
    results: pd.DataFrame
        The diadem search results.
    fasta_path : PathLike
        The FASTA file that was used for the search.
    ms_data_path : PathLike
        The mass spectrometry data file that was searched.

    Returns
    -------
    pd.DataFrame
        The input DataFrame for mokapot.
    """
    # Keep only rank 1 peptides
    results = results.loc[results["rank"] == 1, :]

    # Remove all list columns
    non_list_cols = [c for c in results.columns if not isinstance(results[c][0], list)]
    results = results.loc[:, non_list_cols].drop(columns="rank")

    results["filename"] = Path(ms_data_path).stem
    results["target_pair"] = results["peptide"]
    results.loc[results["decoy"], "target_pair"] = results.loc[
        results["decoy"],
        "peptide",
    ].apply(_decoy_to_target)
    results["decoy"] = ~results["decoy"]
    stripped_peptides = results["target_pair"].str.replace("\\[.*?\\]", "", regex=True)
    results["peptide_length"] = stripped_peptides.str.len()

    # Add the pct-features
    npeak_cols = [x for x in results.columns if "npeaks" in x]
    for x in npeak_cols:
        results[f"{x}_pct"] = 100 * results[x] / results["peptide_length"]

    # Get proteins, although not enirely necessary:
    ## For decoys, this is the corresponding target protein.
    results["proteins"] = stripped_peptides.apply(
        _get_proteins,
        ngram=ProteinNGram.from_fasta(fasta_path, progress=False),
    )

    # Rename columns to the expected values
    expected_names = {"decoy": "is_target"}
    results.rename(columns=expected_names, inplace=True)
    return results


def _decoy_to_target(seq: str, permutation: Iterable[int] | None = None) -> str:
    """Get the target sequence for a decoy peptide.

    Parameters
    ----------
    seq : str
        The decoy peptide sequence with Proforma-style modifications.
    permutation : Sequence[int] | None
        The permuation that was used to generate the decoy from the target sequence.
        If ``None,`` it is assumed to be reversed between the termini.

    Returns
    -------
    str
        The target sequence that generated the decoy sequence.
    """
    seq = re.split(r"(?=[A-Z])", seq)[1:]
    if permutation is None:
        inverted = list(range(len(seq)))
        inverted[1:-1] = reversed(inverted[1:-1])
    else:
        inverted = [None] * len(seq)
        for decoy_idx, target_idx in enumerate(permutation):
            inverted[target_idx] = decoy_idx

    return "".join([seq[i] for i in inverted])


def _get_proteins(peptide: str, ngram: ProteinNGram) -> str:
    """Get the protein(s) that may have generated a peptide.

    Parameters
    ----------
    peptide : str
        The stripped peptide sequence.
    ngram : ProteinNGram
        The n-gram object to look-up sequences.

    Returns
    -------
    str
        The protein or proteins delimited by semi-colons.
    """
    return ";".join(ngram.search_ngram(peptide))
