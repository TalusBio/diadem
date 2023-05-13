"""Aggregate diadem search results."""
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import polars as pl
from mokapot import LinearPsmDataset

from diadem.aggregate.imputers import MFImputer
from diadem.config import DiademConfig


def searches(
    scores: Iterable[pl.DataFrame | pl.LazyFrame | PathLike],
    config: DiademConfig,
) -> tuple[Path]:
    """Aggregate search results and align retention times.

    Parameters
    ----------
    scores : Iterable[PathLike]
        The run-level mokapot parquet files to read.
    fasta_file : Pathlike
        The FASTA file used for the database search.
    config : DiademConfig
        The configuration options.

    Returns
    -------
    peptides : Path
        The accepted peptides
    proteins : Path
        The accepted proteins
    """
    agg = SearchAggregator(scores, config)
    return agg.peptide_path, agg.protein_path


class SearchAggregator:
    """Aggregate search results and align retention times.

    Parameters
    ----------
    scores : Iterable[PathLike]
        The run-level mokapot parquet files to read.
    fasta_file : Pathlike
        The FASTA file used for the database search.
    config : DiademConfig
        The configuration options.
    """

    def __init__(
        self,
        scores: Iterable[PathLike],
        config: DiademConfig,
    ) -> None:
        """Initialize the search aggregator."""
        self.scores = scores
        self.config = config

        base_path = Path(self.config.output_dir) / "aggregate"
        self.peptide_path = base_path / "diadem.search.peptides.parquet"
        self.protein_path = base_path / "diadem.search.proteins.parquet"

        if not self.config.overwrite:
            base_msg = "%s already exists and overwrite is disabled."
            if self.peptide_path.exists():
                raise RuntimeError(base_msg.format(str(self.peptide_path)))
            if self.protein_path.exists():
                raise RuntimeError(base_msg.format(str(self.protein_path)))

        self._peptides = None
        self._proteins = None
        self._ret_time = None
        self._ion_mobility = None

        if len(self.scores) < 2:
            raise ValueError("At least two search results must be provided.")

        # 1. Compute global FDR
        self.assign_confidence()

        # 2. Gather RT/IM for accepted peptides:
        self.collect_imputer_data()

        # 3. Impute RT/IM for missing peptides in each run:
        self.impute()

        # 4. Save the results.
        self.save()

    def assign_confidence(self) -> None:
        """Assign confidence across all runs."""
        keep_cols = ["peptide", "target_pair", "is_target", "mokapot score"]
        try:
            score_df = (
                pl.concat(self.scores, how="vertical")
                .lazy()
                .select(keep_cols)
                .collect()
            )
        except TypeError:
            score_df = pl.concat(
                [pl.read_parquet(s, columns=keep_cols) for s in self.scores],
                how="vertical",
            )

        peptides = LinearPsmDataset(
            psms=(score_df.with_columns(pl.lit("").alias("proteins")).to_pandas()),
            target_column="is_target",
            spectrum_columns="target_pair",
            peptide_column="peptide",
            protein_column="protein",
            feature_columns="mokapot score",
            copy_data=False,
        )

        peptides.add_proteins(
            self.config.fasta_file,
            enzyme=self.config.db_enzyme,
            missed_cleavages=self.config.db_max_missed_cleavages,
            min_length=self.config.peptide_length_range[0],
            max_length=self.config.peptide_length_range[1],
        )

        # Global FDR:
        results = peptides.assign_confidence(
            "mokapot score",
            eval_fdr=self.config.eval_fdr,
            desc=True,
        )

        self._peptides = (
            pl.DataFrame(results.peptides)
            .filter(pl.col("mokapot q-value") <= self.config.eval_fdr)
            .drop("target_pair")
        )

        self._proteins = pl.DataFrame(results.proteins).filter(
            pl.col("mokapot q-value") <= self.config.eval_fdr,
        )

    def collect_imputer_data(self) -> None:
        """Filter run results for confident peptides."""
        keep_cols = ["peptide", "filename", "mokapot q-value"]
        rt_df = []
        im_df = []
        for run in self.scores:
            try:
                run_df = pl.read_parquet(run, columns=keep_cols).lazy()
            except TypeError:
                run_df = run.select(keep_cols).lazy()

            run_df = run_df.filter(
                pl.col("mokapot q-value") <= self.config.eval_fdr,
            ).drop("mokapot q-value")

            fname = run_df["filename"][0]

            # Join with accepted peptides, maintaining order.
            run_df = (
                self._peptides.lazy()
                .join(run_df, how="left", on="peptide")
                .drop(
                    [
                        "mokapot q-value",
                        "mokapot PEP",
                        "mokapot score",
                        "filename",
                        "peptide",
                    ],
                )
                .collect()
            )

            rt_df.append(
                run_df.select(pl.col("RetentionTime").alias(f"retention_time_{fname}")),
            )
            try:
                im_df.append(
                    run_df.select(pl.col("IonMobility").alias(f"ion_mobility_{fname}")),
                )
            except pl.exceptions.ColumnNotFoundError:
                pass

        rt_df = pl.concat(rt_df, how="diagonal")
        self._ret_time = rt_df
        if im_df:
            im_df = pl.concat(im_df, how="diagonal")
            self._ion_mobility = im_df

    def impute(self) -> None:
        """Imput missing retention times and ion mobility values."""
        rt_mat = (
            MFImputer(rng=self.config.seed, task="retention time")
            .search_factors(self._ret_time.to_numpy(), [2, 4, 8, 16])
            .fit_transform(self._ret_time.to_numpy())
        )

        self._ret_time = pl.DataFrame(rt_mat, schema=self._ret_time.columns)

        if self._ion_mobility is not None:
            im_mat = (
                MFImputer(rng=self.config.seed, task="ion mobility")
                .search_factors(self._ion_mobility.to_numpy(), [2, 4, 8, 16])
                .fit_transform(self._ion_mobility.to_numpy())
            )

            self._ion_mobility = pl.DataFrame(
                im_mat,
                schema=self._ion_mobility.columns,
            )

    def save(self) -> tuple(Path):
        """Save the aggregated results."""
        self.peptide_path.parent.mkdir(exist_ok=True)
        pep_dfs = [self._peptides, self._ret_time, self._ion_mobility]
        pl.concat(pep_dfs, how="horizontally").write_parquet(self.peptide_path)
        self._proteins.write_parquet(self.protein_path)
