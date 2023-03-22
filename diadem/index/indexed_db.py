from __future__ import annotations

import copy
from collections import namedtuple
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from fastparquet import write as write_parquet
from loguru import logger
from ms2ml import Config, Peptide
from ms2ml.data.adapters import FastaAdapter
from ms2ml.utils.mz_utils import get_tolerance, sort_all
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm.auto import tqdm

from diadem.config import DiademConfig
from diadem.index.caching import file_cache_dir
from diadem.index.fragment_buckets import (
    FragmentBucket,
    FragmentBucketList,
    PrefilteredMS1BucketList,
)
from diadem.utils import disabled_gc, is_sorted, make_decoy

# Pre-calculating factorials so I do not need
# to calculate them repeatedly while scoring
LOG_FACTORIALS = np.cumsum(np.log(np.arange(1, 1000)))

# TODO consider making everything a float16
# NOTE if i was to always use charge 1 fragments
# then i could use only the neutral mass for indexing ...

DEFAULT_CONFIG = DiademConfig()

# TODO add this to the config
PRELIM_FILTER = 200


def glimpse_array(arr: NDArray, name: str = None) -> str:
    """Utility function to get simple metrics on an array."""
    return f"{name}: min={arr.min()}, max={arr.max()}, shape={arr.shape}"


@lru_cache(1)
def _make_score_dict(ions: str) -> dict[str, dict[str, float | list[float]]]:
    """Internal function that generates dictionary to help with scoring."""
    out = {
        i: {
            "intensities": 0.0,
            "npeaks": 0,
            "mzs": [],
            "mass_errors": [],
        }
        for i in ions
    }
    return out


SeqProperties = namedtuple(
    "SeqProperties",
    "fragments, ion_series, prec_mz, proforma_seq, num_frags",
)


class PeptideScore:
    """Accumulates elements to calculate the score for a peptide."""

    __slots__ = ("id", "ions", "partial_scores", "tot_peaks", "peak_ids")

    def __init__(self, id: int, ions: str) -> None:
        """Accumulates elements to calculate the score for a peptide.

        Parameters
        ----------
        id : int
            The id of the peptide
        ions : str
            The ion series to score (usually "by")

        Examples
        --------
        >>> score = PeptideScore(1, "by")
        >>> score.add_peak("y", mz=234.22, intensity=100, error=0.01, peak_id=1)
        >>> score.add_peak("y", mz=534.22, intensity=200, error=0.012, peak_id=2)
        >>> outs = score.as_row_entry()
        >>> [x for x in outs]  # doctest: +NORMALIZE_WHITESPACE
        ['id', 'b_intensity', 'b_npeaks', 'b_mzs', 'y_intensity', \
         'y_npeaks', 'y_mzs', 'log_intensity_sums', 'log_factorial_peak_sum', \
         'mzs', 'mass_errors', 'avg_abs_dm', 'med_abs_dm', 'spec_indices']
        """  # noqa
        self.id: int = id
        self.ions = ions
        self.partial_scores = copy.deepcopy(_make_score_dict(ions))
        self.tot_peaks = 0
        self.peak_ids = []

    def add_peak(
        self,
        ion: str,
        mz: float,
        intensity: float,
        error: float,
        peak_id: int,
    ) -> None:
        """Adds a peak to the partial score.

        Check the class docstring for more details.

        Parameters
        ----------
        ion : str
            The ion type (e.g. "y").
        mz : float
            The m/z of the peak.
        intensity : float
            The intensity of the peak to add.
        error : float
            The mass error of the peak to add.
        peak_id: int
            Id of the peak. It is usefull to track what peaks within
            a spectrum were a match.
        """
        self.partial_scores[ion]["intensities"] += intensity
        self.partial_scores[ion]["npeaks"] += 1
        self.partial_scores[ion]["mzs"].append(mz)
        self.partial_scores[ion]["mass_errors"].append(error)
        self.peak_ids.append(peak_id)
        self.tot_peaks += 1

    def as_row_entry(self) -> dict[str, float | list[float] | int]:
        """Returns a dictionary of the partial scores.

        The output is meant to be used as a row entry for a pandas dataframe.
        See the class docstring for details and usage.
        """
        out = {}
        out["id"] = self.id

        logfact_peaks = 0
        logints = 0
        mzs = []
        mass_errors = []
        for ion in self.ions:
            out[f"{ion}_intensity"] = self.partial_scores[ion]["intensities"]
            out[f"{ion}_npeaks"] = self.partial_scores[ion]["npeaks"]
            out[f"{ion}_mzs"] = self.partial_scores[ion]["mzs"]
            logfact_peaks += LOG_FACTORIALS[self.partial_scores[ion]["npeaks"]]
            logints += np.log1p(self.partial_scores[ion]["intensities"])
            mzs.extend(self.partial_scores[ion]["mzs"])
            mass_errors.extend(self.partial_scores[ion]["mass_errors"])

        out["log_intensity_sums"] = logints
        out["log_factorial_peak_sum"] = sum(
            self.partial_scores[ion]["npeaks"] for ion in self.ions
        )
        out["mzs"] = mzs
        out["mass_errors"] = mass_errors
        out["avg_abs_dm"] = np.abs(np.array(mass_errors)).mean()
        out["med_abs_dm"] = np.median(np.abs(np.array(mass_errors)))
        out["spec_indices"] = self.peak_ids
        return out


class IndexedDb:
    """IndexedDb class.

    This class is used to create a database of peptides and their
    associated fragments. It is used to score spectra against the
    database.

    There are several ways of populating the database with targets peptides.
    1. from a fasta file using IndexedDb.targets_from_fasta
    2. from a cached sqlite using IndexedDB.index_from_sqlite
    3. from raw arrays using IndexedDB.index_from_arrays
    """

    def __init__(
        self,
        chunksize: int,
        config: DiademConfig = DEFAULT_CONFIG,
        name: str = "db",
    ) -> None:
        """Creates a new IndexedDb object.

        Parameters
        ----------
        chunksize : int
            The chunksize to use for the database. This is the number of
            ions each bucket will store.
        config : DiademConfig, optional
            The configuration to use for the database, by default DEFAULT_CONFIG
        name : str, optional
            The name of the database, by default "db".
            It will be used to label some of the output.

        """
        self.name = name
        self.config = config
        self.chunksize = chunksize
        self.score_nums = 0
        self.empty_slices = 0
        self.prefiltered_ms1 = False
        # CHUNKSIZE = 32 # sage uses 32768 for real world stuff

    @property
    def ms2ml_config(self) -> Config:
        """Extracts an ms2ml config from the internal diadem config."""
        if not hasattr(self, "_ms2ml_config"):
            self._ms2ml_config = self.config.ms2ml_config
        return self._ms2ml_config

    @property
    def targets(self) -> list[Peptide]:
        """Returns a list of peptide objects for the targets in the database."""
        if hasattr(self, "_targets") and self._targets is not None:
            return self._targets
        else:
            raise ValueError("Targets are not set yet in this database")

    @targets.setter
    def targets(self, value: list[Peptide]) -> None:
        """Sets the targets for the database.

        Parameters
        ----------
        value : list[Peptide]
            A list of peptide objects to set as the targets for the database.
        """
        seen = set()
        use = []
        # Since right now my peptide implementation is not hashable,
        # this needs to be done to make sure all targets are unique.
        for x in value:
            x.config = self.ms2ml_config
            if (proforma := x.to_proforma()) in seen:
                continue
            else:
                use.append(x)
                seen.add(proforma)

        self._targets = use
        self._decoys = None
        self.target_proforma = {x.to_proforma() for x in use}

    @property
    def decoys(self) -> list[Peptide]:
        """Returns a list of peptide objects for the decoys in the database.

        If decoys have not been generated yet, they are generated and cached.
        """
        if hasattr(self, "_decoys") and self._decoys is not None:
            pass
        else:
            targets = self.targets
            self._decoys = [
                make_decoy(x) for x in tqdm(targets, desc="Generating Decoys")
            ]
            logger.info(
                (
                    f"Generating database with {len(self.decoys)} decoys,"
                    f" and {len(self.targets)} targets"
                ),
            )

        return self._decoys

    def targets_from_fasta(self, fasta_path: Path | str) -> None:
        """Populates the database with targets from a fasta file.

        Note that the configuration required to digest and get the peptides
        from the fasta are in the config, which is set when initializing the
        database.

        Parameters
        ----------
        fasta_path : Path | str
            The path to the fasta file to use to populate the database.

        """

        def ms1_filter(pep: Peptide) -> Peptide | None:
            mz = pep.mz
            if (mz < pep.config.peptide_mz_range[0]) or (
                mz > pep.config.peptide_mz_range[1]
            ):
                return None
            else:
                return pep

        adapter = FastaAdapter(
            file=fasta_path,
            config=self.config.ms2ml_config,
            only_unique=True,
            enzyme=self.config.db_enzyme,
            missed_cleavages=self.config.db_max_missed_cleavages,
            allow_modifications=False,
            out_hook=ms1_filter,
        )
        sequences = list(adapter.parse())
        assert len(sequences) == len({x.to_proforma() for x in sequences})
        self.targets = sequences

    def prefilter_ms1(
        self,
        ms1_range: tuple[float, float],
        num_decimals: int = 3,
    ) -> IndexedDb:
        """Prefilters the database.

        The filtered database will include peptides fragments a given
        MS1 range.

        Parameters
        ----------
        ms1_range : tuple[float, float]
            The MS1 range to filter the database to.
        num_decimals : int
            The number of decimal places that the fragments will be rounded to.
            This is used for indexing purposes (smaller number = coarser index)
            but not for scoring.


        Returns
        -------
        IndexedDb
            A copy of the database including only the fragments that
            match the provided m/z.
        """
        logger.info(f"Filtering ms1 ranges {ms1_range} in database {self.name}")
        out = copy.copy(self)

        out.bucketlist = self.bucketlist.prefilter_ms1(
            *ms1_range,
            num_decimals=num_decimals,
        )
        out.prefiltered_ms1 = True
        out.seq_prec_mzs = self.seq_prec_mzs
        out.seqs = self.seqs
        out.seq_ids = self.seq_ids

        return out

    def generate_to_parquet(self, dir: Path | str) -> None:
        """Generates the fragments for the targets and decoys into parquet files.

        It uses the targets and decoys stored in `self.targets` and `self.decoys`
        to generate theoretical fragments (based on the `self.config` object).
        And writes them to a pair of parquet files in the passed directory.
        """
        dir = Path(dir)
        seq_file_path = dir / "seqs.parquet"
        frag_file_path = dir / "frags.parquet"

        last_id = self._dump_peptides_parquet(
            seq_file_path=seq_file_path,
            fragment_file_path=frag_file_path,
            peptides=self.targets,
            name="Targets",
            decoy=False,
        )
        self._dump_peptides_parquet(
            seq_file_path=seq_file_path,
            fragment_file_path=frag_file_path,
            peptides=self.decoys,
            name="Decoys",
            decoy=True,
            start_id=last_id + 1,
        )

    # @profile
    def index_from_parquet(self, dir: Path | str) -> None:
        """Generates as index from the passed directory of parquet files.

        See Also
        --------
        self.generate_to_parquet
        """
        dir = Path(dir)
        seq_file = dir / "seqs.parquet"
        frag_file = dir / "frags.parquet"

        seqs_df = pd.read_parquet(seq_file)

        self.seq_prec_mzs = seqs_df["seq_mz"].values
        self.seqs = seqs_df["seq_proforma"].values

        self.target_proforma = set(
            (seqs_df["seq_proforma"][np.invert(seqs_df["decoy"])]).values,
        )

        frags_df = pd.read_parquet(
            frag_file,
        )
        frags_df = frags_df[frags_df["mz"] > self.config.ion_mz_range[0]]
        frags_df = frags_df[frags_df["mz"] < self.config.ion_mz_range[1]]

        self.target_proforma = set(
            (seqs_df["seq_proforma"][np.invert(seqs_df["decoy"])]).values,
        )
        self.index_from_arrays(
            frags_df["mz"].values,
            frag_series=frags_df["ion_series"].values,
            frag_to_prec_ids=frags_df["seq_id"].values,
            prec_mzs=seqs_df["seq_mz"].values,
            prec_seqs=seqs_df["seq_proforma"].values,
        )

    # @profile
    def _dump_peptides_parquet(
        self,
        seq_file_path: Path,
        fragment_file_path: Path,
        peptides: list[Peptide],
        name: str,
        decoy: bool = False,
        start_id: int = 0,
    ) -> int:
        """Inserts peptide properties and fragments to a database.

        Meant to be an internal method.

        Returns
        -------
        int: The id of the last inserted peptide, meant to
             be used as start_id for the next batch.
        """
        one_pct = int(len(peptides) / 100)

        iter_seqs = tqdm(
            peptides,
            desc=f"Generating sequence ions for {name} into database",
            miniters=one_pct,
        )

        seq_chunk = {
            "seq_id": [],
            "seq_mz": [],
            "seq_proforma": [],
            "decoy": [],
        }
        frag_chunk = {
            "mz": [],
            "ion_series": [],
            "seq_id": [],
            "precursor_mz": [],
        }

        append = False
        if seq_file_path.exists():
            append = True
        for seq_id, (frag_mzs, ion_series, prec_mzs, prec_seqs, num_frags) in enumerate(
            (self.seq_properties(x) for x in iter_seqs),
            start=start_id,
        ):
            seq_chunk["seq_id"].append(seq_id)
            seq_chunk["seq_mz"].append(prec_mzs)
            seq_chunk["seq_proforma"].append(prec_seqs)
            seq_chunk["decoy"].append(decoy)

            for w, x, y, z in zip(
                [prec_mzs] * num_frags,
                frag_mzs,
                ion_series,
                [seq_id] * num_frags,
            ):
                frag_chunk["precursor_mz"].append(float(w))
                frag_chunk["mz"].append(float(x))
                frag_chunk["ion_series"].append(y)
                frag_chunk["seq_id"].append(z)

            if seq_id % one_pct == 0:
                if seq_file_path.exists():
                    append = True
                write_parquet(seq_file_path, pd.DataFrame(seq_chunk), append=append)
                write_parquet(
                    fragment_file_path,
                    pd.DataFrame(frag_chunk),
                    append=append,
                )

                # This just flushes the chunk so next iteration starts
                # clean.
                for x in seq_chunk:
                    seq_chunk[x] = []
                for x in frag_chunk:
                    frag_chunk[x] = []

        if len(frag_chunk["mz"]):
            write_parquet(seq_file_path, pd.DataFrame(seq_chunk), append=append)
            write_parquet(fragment_file_path, pd.DataFrame(frag_chunk), append=append)

        return seq_id

    # @profile
    def index_from_sequences(self) -> None:
        """Generates the index from the sequences in the database.

        Usage
        -----
        ```
            db = IndexedDb(...)
            db.targets = [Peptide(...), Peptide(...)]
            or
            db.targets_from_fasta(...)
            db.index_from_sequences()
        ```

        Note:
        ----
          - This requires adding the targets to the database prior to
            calling this function.
        """
        sequences = self.targets + self.decoys
        one_pct = int(len(sequences) / 100)

        with disabled_gc():
            # TODO change the flow here so the properties are added to pre-made lists
            # instead of having to extract the elements after ...
            iter_seqs = tqdm(
                sequences,
                desc=f"Generating sequence ions for {self.name}",
                miniters=one_pct,
            )

            frag_mzs, frag_series, prec_mzs, prec_seqs, num_frags = zip(
                *(self.seq_properties(x) for x in iter_seqs),
            )

            # NOTE: Changing to float16 does not give the correct result
            prec_mzs = np.array(prec_mzs, dtype="float32")
            prec_seqs = np.array(prec_seqs, dtype="object")
            frag_mzs = np.concatenate(list(frag_mzs)).astype("float32")
            frag_series = np.concatenate(list(frag_series))
            seq_ids = np.empty_like(frag_mzs, dtype=int)

            start_position = 0
            for i, n in enumerate(num_frags):
                end_position = n + start_position
                seq_ids[start_position:end_position] = i
                start_position = end_position

            assert start_position == len(seq_ids), (
                f"Ending position of the unique ids {start_position} is not the same as"
                f" the length of the fragment array {len(seq_ids)}"
            )

        self.index_from_arrays(
            frag_mzs=frag_mzs,
            frag_series=frag_series,
            frag_to_prec_ids=seq_ids,
            prec_mzs=prec_mzs,
            prec_seqs=prec_seqs,
        )

    def index_from_arrays(
        self,
        frag_mzs: NDArray[np.float32],
        frag_series: NDArray[np.str],
        frag_to_prec_ids: NDArray[np.int64],
        prec_mzs: NDArray[np.float32],
        prec_seqs: NDArray[np.str],
        prec_ids: None | NDArray[np.int64] = None,
    ) -> None:
        """Generates an index of fragmnents from a series of arrays.

        Parameters
        ----------
        frag_mzs : NDArray[np.float32]
            An array of fragment m/z values.
        frag_series : NDArray[np.str]
            An array of fragment ion series.
        frag_to_prec_ids : NDArray[np.int64]
            An array of sequence ids. (unique identifier of a peptide sequence)
        prec_mzs : NDArray[np.float32]
            An array of precursor m/z values.
        prec_seqs : NDArray[np.str]
            An array of precursor sequences.
        prec_ids: NDArray[np.int64]
            An array with the sequence ids of the precursors.


        Notes
        -----
        The dimensions of the frag_mzs, frag_series and frag_to_prec_ids need
        to be the same.
        And the dimensions of the prec_mzs, prec_seqs and prec_ids
        also have to be the same.

        """
        if not len(prec_mzs) == len(prec_seqs):
            raise ValueError(
                (
                    "The length of the precursor mzs and the precursor sequences needs"
                    " to be the same."
                ),
            )
        if not all(len(frag_mzs) == len(x) for x in [frag_series, frag_to_prec_ids]):
            raise ValueError(
                (
                    "The length of the frag_mz, frag_series and frag_to_prec_ids need"
                    " to be the same"
                ),
            )
        if not len(prec_seqs) == len(np.unique(prec_seqs)):
            raise ValueError("All precursor sequences need to be unique!")

        # Sorted externally by ms2 mz
        with disabled_gc():
            logger.debug(
                f"Sorting by ms2 mz. {frag_mzs.size} total fragments (if needed)",
            )
            if not is_sorted(frag_mzs):
                sorted_frags, sorted_frag_series, sorted_seq_ids = sort_all(
                    frag_mzs,
                    frag_series,
                    frag_to_prec_ids,
                )
                logger.debug("Done sorting (and GC), generating bucketlists")
            else:
                logger.debug("Skippping sortinb because it is already sorted.")
                sorted_frags, sorted_frag_series, sorted_seq_ids = (
                    frag_mzs,
                    frag_series,
                    frag_to_prec_ids,
                )

            del frag_mzs, frag_to_prec_ids, frag_series

        self.bucketlist = FragmentBucketList.from_arrays(
            fragment_mzs=sorted_frags,
            fragment_series=sorted_frag_series,
            precursor_ids=sorted_seq_ids,
            precursor_mzs=prec_mzs[sorted_seq_ids],
            chunksize=self.chunksize,
            sorting_level="ms2",
            been_sorted=True,
        )

        self.seq_prec_mzs = prec_mzs
        self.seqs = prec_seqs
        if prec_ids is None:
            self.seq_ids = np.arange(len(prec_seqs))
        else:
            self.seq_ids = prec_ids

    # @profile
    def seq_properties(self, x: Peptide) -> SeqProperties:
        """Internal method that extracts the peptide properties to build the index."""
        masses = {
            k: np.concatenate(
                [x.ion_series(ion_type=k, charge=c) for c in x.config.ion_charges],
            )
            for k in x.config.ion_series
        }

        ion_series = np.concatenate(
            [np.full_like(v, k, dtype=str) for k, v in masses.items()],
        )
        masses = np.concatenate(list(masses.values()))
        # TODO move this to the config ...
        mass_mask = (masses > 150) * (masses < 2000)
        masses = masses[mass_mask]
        ion_series = ion_series[mass_mask]
        out = SeqProperties(masses, ion_series, x.mz, x.to_proforma(), len(masses))
        return out

    # @profile
    def yield_candidates(
        self,
        ms2_range: tuple[float, float],
        ms1_range: tuple[float, float],
    ) -> Iterator[tuple[int, float, str]]:
        """Yields candidate fragments that match both an ms1 and an ms2 range.

        Nore: MS1 range is ignored when the database has been pre-filtered.

        Parameters
        ----------
        ms2_range : tuple[float, float]
            The ms2 range to search for.
        ms1_range : tuple[float, float]
            The ms2 range to search for.


        """
        yield from self.bucketlist.yield_candidates(ms2_range, ms1_range)

    # @profile
    def score_arrays(
        self,
        precursor_mz: float | tuple[float, float],
        spec_mz: Iterable[float],
        spec_int: Iterable[float],
    ) -> DataFrame | None:
        """Scores a spectrum against the index.

        The result is a data frame containing all generic data required to
        generate a score. Such as the numbe rof peaks per ion series
        and the intensities.

        Parameters
        ----------
        precursor_mz : float | tuple[float, float]
            The precursor m/z value or a tuple of the lower and upper bound
            of the precursor m/z.
        spec_mz : Iterable[float]
            The m/z values of the spectrum.
        spec_int : Iterable[float]
            The intensity values of the spectrum.
        """
        # TDOO: make this a config option
        MIN_PEAKS = 3  # noqa
        self.score_nums += 1

        if hasattr(precursor_mz, "__len__"):
            if len(precursor_mz) == 2:
                ms1_range = precursor_mz
            else:
                raise ValueError(
                    "precursor_mz has to be of length 2 or a single number",
                )
        else:
            ms1_tol = get_tolerance(
                self.config.g_tolerances[0],
                theoretical=precursor_mz,
                unit=self.config.g_tolerance_units[0],
            )
            ms1_range = (precursor_mz - ms1_tol, precursor_mz + ms1_tol)

        peaks = []

        for i, (fragment_mz, fragment_intensity) in enumerate(zip(spec_mz, spec_int)):
            ms2_tol = get_tolerance(
                self.config.g_tolerances[1],
                theoretical=fragment_mz,
                unit=self.config.g_tolerance_units[1],
            )

            candidates = self.yield_candidates(
                ms1_range=ms1_range,
                ms2_range=(fragment_mz - ms2_tol, fragment_mz + ms2_tol),
            )

            for seq, frag, series in candidates:
                # Should tolerances be checked here?
                # IN THEORY, they should have been filtered in the past.
                dm = frag - fragment_mz
                if abs(dm) <= ms2_tol:
                    peaks.append(
                        {
                            "seq": seq,
                            "ion": series,
                            "mz": fragment_mz,
                            "intensity": fragment_intensity,
                            "error": dm,
                            "peak_id": i,
                        },
                    )

        peptide_ids = np.array([x["seq"] for x in peaks])
        ids, counts = np.unique(peptide_ids, return_counts=True)
        gt_min = counts > MIN_PEAKS
        ids, counts = ids[gt_min], counts[gt_min]

        # Just makes sure we dont access more elements than we have
        # ie: top 200 out of 100 candidates.
        top_n_filter = min(PRELIM_FILTER, len(counts))
        indices_top = np.argpartition(counts, -top_n_filter)[-top_n_filter:]
        keep_ids = ids[indices_top]

        scores = {id: PeptideScore(id, self.config.ion_series) for id in keep_ids}
        for peak in peaks:
            pep_id = peak.pop("seq")
            if pep_id in scores:
                scores[pep_id].add_peak(**peak)

        scores = list(scores.values())
        if not scores:
            return None

        scores_df = pd.DataFrame.from_records([x.as_row_entry() for x in scores])
        return scores_df

    # @profile
    def hyperscore(
        self,
        precursor_mz: float | tuple[float, float],
        spec_mz: NDArray[np.float32],
        spec_int: NDArray[np.float32],
        top_n: int = 100,
    ) -> DataFrame:
        """Score a spectrum against the index.

        Parameters
        ----------
        precursor_mz : float | tuple[float, float]
            The precursor m/z of the spectrum. If a tuple is given, the first
            value is the lower bound and the second value is the upper bound.
            If it is a float it will generate a range centered in the passed value
            with the tolerance in the configuration.
        spec_mz : NDArray[np.float32]
            The m/z values of the spectrum.
        spec_int : NDArray[np.float32]
            The intensity values of the spectrum.
        top_n : int, optional
            The number of top scoring peptides to return, by default 100

        Returns
        -------
        DataFrame
            A dataframe with the top scoring peptides.
        """
        assert len(self.seq_ids) == len(self.seqs)
        assert len(self.seq_prec_mzs) == len(self.seqs)
        scores = self.score_arrays(
            precursor_mz=precursor_mz,
            spec_mz=spec_mz,
            spec_int=spec_int,
        )

        if scores is None or len(scores) == 0:
            return None

        scores["Score"] = (
            scores["log_factorial_peak_sum"] + scores["log_intensity_sums"]
        )
        # Calculate requirements for the z score among all other proposed scores!
        score_mean = scores["Score"].mean()
        score_sd = scores["Score"].std()

        # TODO reconsider if I want to do the filtering here.
        # If i dont, I could use the precursor information as a filter ...
        scores = scores.nlargest(top_n, "Score", keep="all")
        scores.sort_values("Score", ascending=False, inplace=True)

        # Calculate the z-score and fill with 0s if the sd is 0.
        scores["ScoreZ"] = (scores["Score"] - score_mean) / score_sd
        scores["ScoreZ"].fillna(0, inplace=True)
        scores["rank"] = [i + 1 for i in range(len(scores))]

        indices_seqs_local = np.searchsorted(self.seq_ids, scores["id"].values)
        assert np.allclose(self.seq_ids[indices_seqs_local], scores["id"].values)

        scores["Peptide"] = self.seqs[indices_seqs_local]
        scores["PrecursorMZ"] = self.seq_prec_mzs[indices_seqs_local]
        scores["decoy"] = [s not in self.target_proforma for s in scores["Peptide"]]
        try:
            assert len(np.unique(scores["Peptide"])) == len(scores), np.unique(
                scores["Peptide"],
                return_counts=True,
            )
        except AssertionError:
            # There is a bug that gets detected here where a single peptide gets
            # scored multiple times ... Usually with different IDs

            # This issue happens when a sequence is also in the decoys
            logger.error(
                (
                    f"{scores} has multiple peptides with the "
                    "same id (ocasionally happens when it is both a "
                    "target and a decoy)"
                ),
            )

        return scores

    # @profile
    def index_prefiltered_from_parquet(
        self,
        cache_path: PathLike,
        min_mz: float,
        max_mz: float,
    ) -> IndexedDb:
        """Generates a pre-filtered index from a parquet cache.

        The parquet cache is a directory with 2 parquet files, the frags.parquet
        and the seqs.parquet.

        This function queries the files and returns a indexed db with only the required
        data.
        """
        frags_df = pl.scan_parquet(str(cache_path) + "/frags.parquet")
        seqs_df = pl.scan_parquet(str(cache_path) + "/seqs.parquet")

        logger.info(
            f"Filtering ms1 ranges {min_mz} to {max_mz} in database {self.name}",
        )

        joint_frags = (
            frags_df.filter(pl.col("precursor_mz") >= min_mz)
            .filter(pl.col("precursor_mz") < max_mz)
            # .unique(maintain_order=False)
            .sort(pl.col("mz"))
            .collect()
        )

        out = copy.copy(self)
        out.bucketlist = PrefilteredMS1BucketList(
            [
                FragmentBucket(
                    fragment_mzs=joint_frags["mz"].to_numpy().astype(np.float32),
                    fragment_series=joint_frags["ion_series"].to_numpy(),
                    precursor_ids=joint_frags["seq_id"].to_numpy(),
                    precursor_mzs=joint_frags["precursor_mz"]
                    .to_numpy()
                    .astype(np.float32),
                    sorting_level="ms2",
                    is_sorted=True,
                ),
            ],
            num_decimal=2,
            max_frag_mz=2000,
            progress=True,
        )

        out.prefiltered_ms1 = True
        # TODO change it so the only required section
        # is the proforma seqs that are in the mz range
        chunk_seq_df = (
            seqs_df.filter(pl.col("seq_mz") >= min_mz)
            .filter(pl.col("seq_mz") < max_mz)
            .unique(maintain_order=False)
            .sort(pl.col("seq_id"))
        ).collect()
        out.seq_prec_mzs = chunk_seq_df["seq_mz"].to_numpy().astype(np.float32)
        out.seqs = chunk_seq_df["seq_proforma"].to_numpy()
        out.seq_ids = chunk_seq_df["seq_id"].to_numpy()

        target_set = chunk_seq_df.select(["seq_proforma", "decoy"])
        target_set = set(
            target_set["seq_proforma"].to_numpy()[
                np.invert(target_set["decoy"].to_numpy())
            ],
        )

        if len(target_set) == 0:
            logger.warning(f"No targets were found in range {min_mz}-{max_mz}")
        out.target_proforma = target_set
        return out


def db_from_fasta(
    fasta: Path | str,
    chunksize: int,
    config: DiademConfig,
    index: bool = True,
) -> tuple[IndexedDb, str]:
    """Created a peak index database from a fasta file.

    It internally checks the existance of a cache in the form of an sqlite file.
    Future implementations will allow cahching in the form of parquet.
    """
    index_config = config.index_config
    config_hash = index_config.hash()
    file_cache = file_cache_dir(file=fasta)
    curr_cache = file_cache / config_hash

    db = IndexedDb(chunksize=chunksize, config=config)
    if not curr_cache.exists() or not (curr_cache / "seqs.parquet").exists():
        logger.info(f"Unable to find cache location {curr_cache}, will generate it.")
        curr_cache.mkdir(parents=True, exist_ok=True)
        db.targets_from_fasta(fasta)
        db.generate_to_parquet(dir=curr_cache)
    else:
        logger.info(f"Found cache location at: {curr_cache}, will use those values.")

    if index:
        db.index_from_parquet(dir=curr_cache)

    return db, curr_cache
