import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str)


def main():
    args = parser.parse_args()
    base_dir = args.base_dir

    # Find the data
    parquet_matches = list(Path(base_dir).glob("*.parquet"))
    peptide_matches = [x for x in parquet_matches if "peptides" in x.name]
    parquet_matches = [x for x in parquet_matches if "peptides" not in x.name]

    # Load the data
    assert len(parquet_matches) == 1
    assert len(peptide_matches) == 1
    parquet_path = parquet_matches[0]
    peptide_matches[0]

    # Plot the data
    foo = pl.scan_parquet(parquet_path)
    df = foo.select(pl.col(["Score", "decoy", "peptide"])).collect()
    bins = np.histogram_bin_edges(df["Score"].to_numpy(), bins=100)
    plt.hist(df.filter(pl.col("decoy"))["Score"], alpha=0.6, label="decoy", bins=bins)
    plt.hist(
        df.filter(pl.col("decoy").is_not())["Score"],
        alpha=0.6,
        label="target",
        bins=bins,
    )
    plt.legend()
    plt.savefig(Path(base_dir) / "score_histogram_psm.png")

    df = df.groupby("peptide").max()
    bins = np.histogram_bin_edges(df["Score"].to_numpy(), bins=100)
    plt.hist(df.filter(pl.col("decoy"))["Score"], alpha=0.6, label="decoy", bins=bins)
    plt.hist(
        df.filter(pl.col("decoy").is_not())["Score"],
        alpha=0.6,
        label="target",
        bins=bins,
    )
    plt.legend()
    plt.savefig(Path(base_dir) / "score_histogram_peptide.png")

    plt.yscale("log")
    plt.savefig(Path(base_dir) / "log_score_histogram_peptide.png")
