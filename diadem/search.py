"""A class to score peptides"""
import itertools
from typing import List, Union
from dataclasses import dataclass

import numpy as np
import numba as nb
from tqdm.auto import tqdm

from . import score_functions
from .feature import Feature
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
                window_mz=window,
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
    window_mz : tuple of float
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
        window_mz,
        width,
        tol,
        open_,
        min_corr,
        rng,
    ):
        """Initialize a BaseWindowScorer"""
        self._run = run
        self._peptides = peptides
        self._window_mz = window_mz
        self._width = width
        self._tol = tol
        self._open = bool(open_)
        self._min_corr = min_corr
        self._rng = np.random.default_rng(rng)

        self._score_function = score_functions.hyperscore
        self._pbar = None
        self._unmatched_peaks = 0

        with self._run:
            self._n_peaks = self._run.n_peaks

    def __iter__(self):
        """Iterate through the features in a run"""
        with self._run:
            self._run.reset()
            while True:
                feature = self._next_peak()
                if feature is None:
                    break

                yield feature

            self._run.reset()

    def search(self):
        """Perform the database search

        Returns
        -------
        list of PeptideScore
            The scored peptides
        """
        scores = []
        desc = (
            f"Searching m/z {self._window_mz[0]:.2f}-"
            f"{self._window_mz[1]:.2f}"
        )
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
        with self._peptides:
            precursors, frags = self._lookup_precursors(query_feature.mean_mz)

        feat_sizes = [len(l) for l in frags]
        frags = itertools.chain.from_iterable(frags)
        found_features = self._run.find_features(
            frags,
            window=self._window_mz,
            ret_time=(
                query_feature.ret_times.min(),
                query_feature.ret_times.max(),
            ),
            tol=self._tol,
        )
        print("found features.")
        areas = []
        for feature in found_features:
            if feature is None:
                areas.append(0.0)

            feature.lower_bound = query_feature.lower_bound
            feature.upper_bound = query_feature.upper_bound
            corr = pearson_corr(query_feature.peak, feature.peak)
            if corr > self._min_corr:
                areas.append(feature.area)
            else:
                feature.append(0.0)

        print("got areas.")

        best_peptide = None
        best_score = -np.inf
        prev_stop = 0
        for (prec, target), n_feat in zip(precursors, feat_sizes):
            pep_area = np.array(areas[prev_stop:n_feat])
            score_val = self._score_function(pep_area)
            if score_val < best_score:
                continue

            if score_val == best_score:
                # Break ties randomly.
                if self._rng.integers(1, endpoint=True):
                    continue

            pep_feat = [query_feature]
            pep_feat += [
                f for f in found_features[prev_stop:n_feat] if f is not None
            ]

            best_peptide = PeptideScore(
                index=prec,
                features=pep_feat,
                score=score_val,
                target=target,
            )
            prev_stop += n_feat

        print("Found best peptide.")

        elim_mz = itertools.chain.from_iterable(
            [f.row_ids for f in best_peptide.features]
        )
        self._run.remove_ions(elim_mz)
        self._pbar.update(len(elim_mz))
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
            prec_mz = self._window_mz

        precursors = self._peptides.fragment_to_precursors(
            fragment_mz,
            self._tol,
            prec_mz,
        )

        prec_idx = [p[0] for p in precursors]
        fragments = self._peptides.precursors_to_fragments(
            prec_idx,
            to_int=True,
        )

        return precursors, fragments

    def _next_peak(self):
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
        top_query = self._run.cur.execute(
            """
            SELECT fi.mz, fr.scan_start_time, MAX(fi.intensity)
                FROM fragment_ions AS fi
            LEFT JOIN fragments AS fr
                ON fi.spectrum_idx=fr.spectrum_idx
            LEFT JOIN used_ions AS ui
                ON fi.ROWID=ui.fragment_idx
            WHERE fr.isolation_window_lower >= ?
                AND fr.isolation_window_upper <= ?
                AND ui.fragment_idx IS NULL
            """,
            self._window_mz,
        )

        frag_mz, ret_time, _ = top_query.fetchone()
        if not frag_mz:
            return None

        min_rt = ret_time - self._width
        max_rt = ret_time + self._width
        feature = self._run.find_features(
            frag_mz,
            self._window_mz,
            (min_rt, max_rt),
        )
        self._run.remove_ions(feature[0].row_ids)
        return feature[0]


@nb.njit
def pearson_corr(x_array, y_array):
    """Calcualte the Pearson correlation between two arrays.

    Parameters
    ----------
    x_array : numpy.ndarray
    y_array : numpy.ndarray
        The arrays to calculate the correlation between.

    Returns
    -------
    float
        The Pearson correlation coefficient.
    """
    x_diffs = x_array - x_array.mean()
    y_diffs = y_array - y_array.mean()
    coeff = np.sum(x_diffs * y_diffs) / np.sqrt(
        np.sum(x_diffs**2) * np.sum(y_diffs**2)
    )
    return coeff
