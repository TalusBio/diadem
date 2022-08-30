"""A class to score peptides"""
import time
from typing import List, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import scoring
from .feature import Feature, extract_feature
from .peptides import PeptideDB
from .msdata import DIARunDB


@dataclass
class PeptideScore:
    """The scores for a precursor"""

    index: int
    features: List[Feature]
    score: float
    target: bool


class PeptideSearcher:
    """Score peptides against features in a the DIA mass spectrometry data.

    Parameters
    ----------
    run : DIARunDB
        The DIA data to use.
    peptides : PeptideDB
        The peptide inverted index to use.
    width : float
        The maximum width of a chromatographic peak in seconds.
    tol : float
        The fragment ion matching tolerance.
    open_ : bool
        Open-modification search?
    min_corr : float, optional
        The minimum correlation for a feature group.
    rng : int or numpy.random.Generator, optional
        The random number generator.
    """

    def __init__(
        self,
        run: DIARunDB,
        peptides: PeptideDB,
        width: float = 1.0,
        tol: float = 10,
        open_: bool = True,
        min_corr: float = 0.9,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """Initialize a PeptideSearcher"""
        self._run = run
        self._peptides = peptides
        self._width = width
        self._tol = tol
        self._open = open_
        self._min_corr = min_corr
        self._rng = np.random.default_rng(rng)

        with self._run:
            self._windows = self._run.windows

    def search(self) -> List[PeptideScore]:
        """Search the data.

        Returns
        -------
        list of PeptideScore objects
            The scored peptides.
        """
        peptide_scores = []
        for window in self._windows:
            scorer = WindowScorer(
                run=self._run,
                peptides=self._peptides,
                window=window,
                width=self._width,
                tol=self._tol,
                open_=self._open,
                min_corr=self._min_corr,
                rng=self._rng.integers(9999),
            )
            peptide_scores += scorer.search()
            break

        return peptide_scores


class WindowScorer:
    """Score peptides against features in the DIA mass spectrometry data.

    Parameters
    ----------
    run : DIARunDB
        The DIA data to use.
    peptides : PeptideDB
        The peptide inverted index to use.
    window : tuple of float
        The m/z boundaries of the DIA window to search.
    width : float
        The maximum width of a chromatographic peak.
    tol : float
        The fragment ion matching tolerance.
    open_ : bool
        Open-modification search?
    min_corr : float, optional
        The minimum correlation for a feature group.
    rng : int of numpy.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        run,
        peptides,
        window,
        width,
        tol,
        open_,
        min_corr,
        rng,
    ):
        """Initialize a BaseWindowScorer"""
        self._run = run
        self._peptides = peptides
        self._window = window
        self._width = width
        self._tol = tol
        self._open = bool(open_)
        self._min_corr = min_corr
        self._rng = np.random.default_rng(rng)

        self._pbar = None
        self._unmatched_peaks = 0

        with self._run:
            self._n_peaks = self._run.n_peaks
            self._peaks = pd.read_sql(
                """
                SELECT fi.* FROM fragment_ions AS fi
                JOIN fragments AS fr
                    ON fi.spectrum_idx=fr.spectrum_idx
                WHERE fr.isolation_window_lower >= ?
                    AND fr.isolation_window_upper <= ?
                """,
                self._run.con,
                params=self._window,
            )
            self._scans = pd.read_sql(
                """
                SELECT spectrum_idx, scan_start_time FROM fragments
                WHERE isolation_window_lower >= ?
                    AND isolation_window_upper <= ?
                """,
                self._run.con,
                params=self._window,
            )

        # Sort by intensity within each spectrum:
        self._peaks = (
            self._peaks.merge(self._scans, how="left")
            .sort_values(
                ["scan_start_time", "intensity"],
                ascending=[True, False],
            )
            .reset_index(drop=True)
        )

        # Indices of all peaks ordered by abundance:
        self._order = list(np.argsort(-self._peaks["intensity"].values))

    def __iter__(self):
        """Iterate through the features in a run"""
        return self

    def __next__(self):
        """Return the next most intense feature.

        Returns
        -------
        mean_mz : int
            The mean integerized m/z of the feature.
        peak_area : float
            The baseline-subtracted peak area of the feature.
        bounds : tuple of int
            The peak boundaries
        idx_bounds : tuple of int
            The spectrum bounds.


        """
        if self._peaks.empty:
            raise StopIteration

        peak = self._peaks.loc[self._order[:1], :]
        query = (int(peak["mz"]), float(peak["scan_start_time"]))
        feat_info, _ = extract_feature(
            *query,
            mz_array=self._peaks["mz"].to_numpy(),
            int_array=self._peaks["intensity"].to_numpy(),
            rt_array=self._peaks["scan_start_time"].to_numpy(),
            mz_tol=self._tol,
            rt_tol=self._width,
        )
        feature = Feature(*query, *feat_info)
        feature.update_bounds()
        return feature

    def search(self):
        """Perform the database search

        Returns
        -------
        list of PeptideScore
            The scored peptides
        """
        scores = []
        desc = f"Searching m/z {self._window[0]:.2f}-" f"{self._window[1]:.2f}"
        self._pbar = tqdm(
            desc=desc,
            unit="peaks",
            total=self._n_peaks,
        )
        for feature in self:
            pfm = self._score_corr_features(feature)
            if pfm is not None:
                scores.append(pfm)

        self._pbar.close()
        return scores

    def _score_corr_features(self, query_feature):
        """Find features that are correlated with the query feature.

        Parameters
        ----------
        fragment_list : list of list of int
            The fragment ions to for each precursor.
        area : float
            The peak area of the query feature.
        bounds : tuple of int
            Indices of the integration bounds for the query feature.
        spectra : tuple of int
            Boundary spectrum indices for the query feature.

        Returns
        -------
        PeptideScore
            The best score precursor for the feature.
        """
        query_mz = int(query_feature.mean_mz)
        rt_bounds = query_feature.ret_time_bounds

        # Get candidates:
        with self._peptides:
            precursors, fragments = self._lookup_precursors(query_mz)

        # Get a smaller list of peaks:
        within_rt = self._peaks["scan_start_time"].between(*rt_bounds)
        peaks = self._peaks.loc[within_rt]
        offset = peaks.index.min()
        mz_array = peaks["mz"].to_numpy()
        int_array = peaks["intensity"].to_numpy()
        rt_array = peaks["scan_start_time"].to_numpy()

        eliminated = set()

        # Find the best scoring peptide:
        best_peptide = None
        best_rows = []
        best_score = -np.inf
        features = []
        for prec, frags in zip(precursors, fragments):
            score_val, feat_info, rows = scoring.score_precursor(
                query_mz=query_mz,
                query_rt=query_feature.query_rt,
                query_peak=query_feature.peak,
                lower_bound=query_feature.lower_bound,
                upper_bound=query_feature.upper_bound,
                frag_array=np.array(frags),
                mz_array=mz_array,
                int_array=int_array,
                rt_array=rt_array,
                mz_tol=self._tol,
                rt_tol=self._width,
                min_corr=self._min_corr,
            )
            if score_val < best_score:
                continue

            if score_val == best_score:
                # Break ties randomly
                if self._rng.integers(1, endpoint=True):
                    continue

            features = [Feature(*f) for f in feat_info]
            best_rows = rows + offset
            best_peptide = PeptideScore(
                index=prec[0],
                features=features,
                score=score_val,
                target=prec[1],
            )

        self._peaks = self._peaks.drop(best_rows)
        best_rows = set(best_rows)
        self._order = [i for i in self._order if i not in best_rows]
        self._pbar.update(len(best_rows))
        return best_peptide

    def _lookup_precursors(self, fragment_mz):
        """Look up the precursor that generated a fragment m/z

        Parameters
        ----------
        fragment_mz : int
            The integerized fragment m/z to look up.

        Return
        ------
        precursors : list of int
            The precursor indices.
        fragments : list of list of int
            The fragment m/z corresponding to each precursor.
        """
        if self._open:
            prec_mz = None
        else:
            prec_mz = self._window

        t1 = time.time()
        precursors = self._peptides.fragment_to_precursors(
            fragment_mz,
            self._tol,
            prec_mz,
        )
        t2 = time.time()

        prec_idx = [p[0] for p in precursors]
        t3 = time.time()
        fragments = self._peptides.precursors_to_fragments(
            prec_idx,
            to_int=True,
        )
        t4 = time.time()
        print(t2 - t1, t4 - t3)

        return precursors, fragments
