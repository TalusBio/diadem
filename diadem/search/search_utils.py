from os import PathLike
from pathlib import Path

from pandas import DataFrame
from pyteomics.proforma import parse

from diadem.index.protein_index import ProteinNGram


def make_pin(
    results: DataFrame,
    fasta_path: PathLike,
    mzml_path: PathLike,
    pin_path: PathLike,
) -> None:
    """Makes a '.pin' file from a dataframe of results.

    It writes the .pin file to disk (`pin_path` argument).
    """
    # Postprocessing of the results dataframe for percolator
    ## 1. keep only rank 1 peptides
    results = results[results["rank"] == 1]

    ## Remove all list columns
    non_list_cols = [c for c in results.columns if not isinstance(results[c][0], list)]
    results = results[non_list_cols]

    ## Add protein names
    ngram = ProteinNGram.from_fasta(str(fasta_path))
    stripped_peptides = [
        "".join([x[0] for x in parse(y)[0]]) for y in results["Peptide"]
    ]
    proteins = [";".join(ngram.search_ngram(x)) for x in stripped_peptides]
    results["Proteins"] = proteins

    ## Add the pct-features
    results["PeptideLength"] = [len(x) for x in stripped_peptides]
    npeak_cols = [x for x in results.columns if "npeaks" in x]
    for x in npeak_cols:
        results[f"{x}_pct"] = 100 * results[x] / results["PeptideLength"]

    ## Convert the decoys column to the right format
    results["decoy"] = [-1 if d else 1 for d in results["decoy"]]

    ## Add a scan number column ....
    # TODO add to the diadem logic to include the representative scan ID
    results["ScanNr"] = list(range(len(results)))
    results["Filename"] = Path(mzml_path).stem

    ## Rename columns to the expected values
    ## Some columns in mokapot/percolator require a specific name
    expected_names = {
        "id": "SpecID",
        "ScanNr": "ScanNr",
        "Peptide": "Peptide",
        "Proteins": "Proteins",
        "decoy": "Label",
    }
    results.rename(columns=expected_names, inplace=True)
    results.to_csv(pin_path, index=False, sep="\t")
