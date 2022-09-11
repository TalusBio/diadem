"""A class to score peptides"""
import time
from typing import List, Union
from dataclasses import dataclass

import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import feature
from . import score_functions
from .feature import Feature
from .index import FragmentIndex
from .msdata import DiaRun


@dataclass
class PeptideScore:
    """The scores for a precursor"""

    seq: str
    mods: List[float]
    features: List[Feature]
    score: float
    is_decoy: bool

    @property
    def peaks(self):
        """Return the peak indices used in the features."""
        peak_idx = np.hstack([f.peaks for f in self.features])
        return peak_idx


class PeptideSearcher:
    """Score peptides against features in a the DIA mass spectrometry data.

    Parameters
    ----------
    run : DiaRun
        The DIA data to use.
    index : FragmentIndex
        The peptide inverted index to use.
    peak_width : float
        The maximum width of a chromatographic peak in minutes.
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
        run: DiaRun,
        index: FragmentIndex,
        peak_width: float = 1.0,
        tol: float = 10,
        open_: bool = True,
        min_corr: float = 0.9,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """Initialize a PeptideSearcher"""
        self._run = run
        self._index = index
        self._peak_width = peak_width
        self._tol = tol
        self._open = open_
        self._min_corr = min_corr
        self._rng = np.random.default_rng(rng)

    def search(self) -> List[PeptideScore]:
        """Search the data.

        Returns
        -------
        list of PeptideScore objects
            The scored peptides.
        """
        peptide_scores = []
        for key, window in self._run.windows.items():
            scorer = WindowScorer(
                window=window,
                index=self._index,
                peak_width=self._peak_width,
                tol=self._tol,
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
    window : DiaWindow
        The DIA isolation window to search.
    index : FragmentIndex
        The fragment index to use.
    peak_width : float
        The maximum width of a chromatographic peak.
    tol : float
        The fragment ion matching tolerance.
    min_corr : float, optional
        The minimum correlation for a feature group.
    rng : int of numpy.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        window,
        index,
        peak_width,
        tol,
        min_corr,
        rng,
    ):
        """Initialize a BaseWindowScorer"""
        self._window = window
        self._index = index
        self._peak_width = peak_width
        self._tol = tol
        self._min_corr = min_corr
        self._rng = np.random.default_rng(rng)

        self._pbar = None
        self._unmatched_peaks = 0

    def search(self):
        """Perform the database search

        Returns
        -------
        list of PeptideScore
            The scored peptides
        """
        scores = []
        with self._window as win:
            pbar = tqdm(unit="peaks", total=len(win))
            for peak in win:
                mz, _, rt = win.peak(peak)
                feat = win.xic(
                    mz,
                    rt - self._peak_width,
                    rt + self._peak_width,
                    self._tol,
                )
                pfm = self._score_peptides(feat)
                win.consume(peak)
                pbar.update(1)
                if pfm is not None:
                    scores.append(pfm)
                    pfm_peaks = pfm.peaks
                    win.consume(pfm_peaks)
                    pbar.update(len(pfm_peaks))

            pbar.close()
        return scores

    def _score_peptides(self, query_feature):
        """Score peptides agains the query feature.

        Parameters
        ----------
        query_feature : Feature
            A query feature.

        Returns
        -------
        PeptideScore
            The best scoring peptide for the feature.
        """
        query_feature.update_bounds()
        query_mz = query_feature.moverz
        rt_bounds = query_feature.ret_time_bounds
        seq_scores, feats_used = [], []
        precursors = self._index.precursors_from_fragment(query_mz, self._tol)
        for seq, mods, is_decoy in precursors:
            feats = [
                self._window.xic(f, *rt_bounds, tol=self._tol)
                for f in self._index.fragments(seq, mods)
            ]
            if not feats:
                continue

            seq_score, used = score(
                query_feature.peak,
                tuple(f.intensity_array for f in feats),
                query_feature.lower_bound,
                query_feature.upper_bound,
                query_feature.rt_array,
                self._min_corr,
                score_functions.mini_hyperscore,
            )

            seq_scores.append(seq_score)
            feats_used.append([f for f, u in zip(feats, used) if u])

        if not seq_scores:
            return None

        winner = np.argmax(np.array(seq_scores))
        return PeptideScore(
            seq,
            mods,
            [query_feature] + feats_used[winner],
            seq_scores[winner],
            is_decoy,
        )


@nb.njit
def score(
    query_peak,
    frag_arrays,
    lower_bound,
    upper_bound,
    rt_array,
    min_corr,
    score_fn,
):
    """Score a peptide against a collection of features.

    Parameters
    ----------
    query_peak : np.ndarray
        The query feature peak.
    frag_arrays : tuple of np.ndarray
        The intensity arrays of the other fragments.
    lower_bound : int
        The lower bound of the integrated query feature.
    upper_bound : int
        The upper bound of the integrated query feature.
    rt_array : np.ndarray
        The retention time array for the feature.
    min_corr : float
        The minimum allowed correlation.
    score_fn : Callable
        The score function to use.

    Returns
    -------
    np.ndarray
        The correlation for each vector.
    """
    x_diffs = query_peak - np.mean(query_peak)
    x_rss = np.sum(x_diffs**2)

    corr = []
    areas = []
    for idx, array in enumerate(frag_arrays):
        if (array > 0).sum() < (0.25 * len(array)):
            corr.append(0)
            areas.append(0)
            continue

        # For the Pearson Correlation:
        peak = feature.calc_peak(array, lower_bound, upper_bound)
        y_diffs = peak - np.mean(peak)
        denom = np.sqrt(x_rss * np.sum(y_diffs**2))
        if not denom:
            corr.append(0)
            areas.append(0)
            continue

        corr.append(np.sum(x_diffs * y_diffs) / denom)
        if corr[idx] < min_corr:
            areas.append(0)
            continue

        areas.append(
            feature.integrate(
                peak,
                rt_array,
                lower_bound,
                upper_bound,
            )
        )

    areas.append(
        feature.integrate(
            query_peak,
            rt_array,
            lower_bound,
            upper_bound,
        )
    )

    areas = np.array(areas)
    corr = np.array(corr)

    score_val = score_fn(areas)
    used = corr >= min_corr
    return score_val, used
