import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import vizta  # First import vizta
from loguru import logger

vizta.mpl.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str)
parser.add_argument(
    "--prefix",
    type=str,
    help=(
        "prefix to use to subset the parquet files, for instance 'hela' will only match"
        " '{base_dir}/hela*.parquet'"
    ),
)


def main():
    args = parser.parse_args()
    base_dir = args.base_dir
    prefix = args.prefix

    # Find the data
    parquet_matches = list(Path(base_dir).glob(f"{prefix}*.parquet"))
    peptide_matches = [x for x in parquet_matches if "peptides" in x.name]
    parquet_matches = [x for x in parquet_matches if "peptides" not in x.name]
    logger.info(f"Found {len(parquet_matches)} parquet files")
    logger.info(f"Found {len(peptide_matches)} peptide files")
    logger.info(f"Found {len(parquet_matches)} parquet files")

    # Load the data
    assert len(parquet_matches) == 1
    assert len(peptide_matches) == 1
    parquet_path = parquet_matches[0]

    # Plot the data
    foo = pl.scan_parquet(parquet_path)
    df = foo.select(pl.col(["Score", "decoy", "peptide"])).collect()
    bins = np.histogram_bin_edges(df["Score"].to_numpy(), bins=100)
    plt.hist(
        df.filter(pl.col("decoy").is_not())["Score"],
        alpha=0.6,
        label="target",
        bins=bins,
        linewidth=0.01,
    )
    plt.hist(
        df.filter(pl.col("decoy"))["Score"],
        alpha=0.6,
        label="decoy",
        bins=bins,
        linewidth=0.01,
    )
    plt.legend()
    plt.title(f"PSM Score Histogram\n{prefix}")
    plt.xlabel("score")
    plt.ylabel("Frequency")
    plt.savefig(Path(base_dir) / f"{prefix}_score_histogram_psm.png")
    plt.clf()

    df = df.groupby("peptide").max()
    bins = np.histogram_bin_edges(df["Score"].to_numpy(), bins=100)
    plt.hist(
        df.filter(pl.col("decoy").is_not())["Score"],
        alpha=0.6,
        label="target",
        bins=bins,
        linewidth=0.01,
    )
    plt.hist(
        df.filter(pl.col("decoy"))["Score"],
        alpha=0.6,
        label="decoy",
        bins=bins,
        linewidth=0.01,
    )
    plt.legend()
    plt.title(f"Peptide Score Histogram\n{prefix}")
    plt.xlabel("score")
    plt.ylabel("Frequency")
    plt.savefig(Path(base_dir) / f"{prefix}_score_histogram_peptide.png")

    plt.yscale("log")
    plt.title(f"Peptide Score Histogram (log scale)\n{prefix}")
    plt.xlabel("log(score)")
    plt.ylabel("Frequency")
    plt.savefig(Path(base_dir) / f"{prefix}_log_score_histogram_peptide.png")
    plt.clf()

    plt.scatter(
        y=df["Score"],
        x=np.arange(len(df)),
        c=["red" if x else "blue" for x in df["decoy"]],
        s=0.5,
        alpha=0.4,
    )
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title(f"Peptide Score Over Iterations\n{prefix}")
    plt.savefig(Path(base_dir) / f"{prefix}_scores_over_time.png")
    plt.clf()

    pep_parquet = pl.scan_parquet(peptide_matches[0])
    qvals = (
        pep_parquet.filter(pl.col("is_target") & (pl.col("mokapot q-value") < 0.05))
        .select(pl.col(["mokapot q-value"]))
        .sort("mokapot q-value")
        .collect()
    )
    plt.plot(qvals, np.arange(len(qvals)))
    plt.title(f"Peptide q-values\n{prefix}")
    plt.savefig(Path(base_dir) / f"{prefix}_peptide_qval.png")
    plt.clf()

    metrics = {}
    metrics["NumPeptides_q_0.01"] = len(qvals.filter(pl.col("mokapot q-value") < 0.01))
    metrics["AvgTargetScore"] = df.filter(pl.col("decoy").is_not())["Score"].mean()
    metrics["TargetQ95Score"] = df.filter(pl.col("decoy").is_not())["Score"].quantile(
        0.95,
    )
    metrics["AvgDecoyScore"] = df.filter(pl.col("decoy"))["Score"].mean()
    metrics["DecoyQ95Score"] = df.filter(pl.col("decoy"))["Score"].quantile(0.95)

    # Write metrics to toml file
    with open(Path(base_dir) / f"{prefix}_metrics.toml", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k} = {v}\n")


if __name__ == "__main__":
    main()
