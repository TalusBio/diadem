"""A class to score peptides"""
import itertools
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import numba as nb
from tqdm.auto import tqdm

from . import score_functions, utils
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

        self._score_function = score_functions.hyperscore
        self._pbar = None
        self._unmatched_peaks = 0

        with self._run:
            self._n_peaks = self._run.n_peaks
            self._peaks = pd.read_sql(
                """
                SELECT fi.ROWID, fi.* FROM fragment_ions AS fi
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

        self._peaks = self._peaks.sort_values(
            "intensity", ascending=False
        ).reset_index(drop=True)

        self._scans = self._scans.sort_values("scan_start_time").reset_index(
            drop=True
        )

    def __iter__(self):
        """Iterate through the features in a run"""
        with self._run:
            self._run.reset()
            while not self._peaks.empty:
                feature = self._next_peak()
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
        with self._peptides:
            precursors = self._lookup_precursors(query_feature.mean_mz)

        rt_bounds = (
            query_feature.ret_times.min(),
            query_feature.ret_times.max(),
        )
        within_rt = self._scans["scan_start_time"].between(*rt_bounds)
        scans = (
            self._scans.loc[within_rt, "spectrum_idx"]
            .reset_index(drop=True)
            .reset_index()
        )
        peaks = self._peaks.merge(scans, how="right")

        best_peptide = None
        best_score = -np.inf
        features = []
        for prec, frags in tqdm(zip(*precursors)):
            prec_features = [query_feature]
            frag_areas = [query_feature.area]
            for frag in frags:
                feat = self._find_feature(
                    feature_mz=frag,
                    peaks=peaks,
                    ret_times=scans["scan_start_time"].values,
                    peak_bounds=(
                        query_feature.lower_bound,
                        query_feature.upper_bound,
                    ),
                )

                if feat is None:
                    continue

                features.append(feat)
                corr = pearson_corr(query_feature.peak, feat.peak)
                if corr > self._min_corr:
                    frag_areas.append(feat.area)
                else:
                    frag_areas.append(0.0)

            score_val = self._score_function(np.array(frag_areas))
            if score_val < best_score:
                continue

            if score_val == best_score:
                # Break ties randomly
                if self._rng.integers(1, endpoint=True):
                    continue

            best_peptide = PeptideScore(
                index=prec[0],
                features=prec_features,
                score=score_val,
                target=prec[1],
            )

        elim_mz = itertools.chain.from_iterable(
            [f.row_ids for f in best_peptide.features]
        )
        self._peaks = self._peaks[~self._peaks["rowid"].isin(elim_mz)]
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
            prec_mz = self._window

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

    def _find_feature(
        self,
        feature_mz: int,
        peaks: pd.DataFrame,
        ret_times: np.ndarray,
        peak_bounds: Optional[Tuple[float, float]] = None,
    ):
        """Extract a feature from candidate peaks.

        Parameters
        ----------
        feature_mz : int
            The integerized m/z of a feature.
        peaks : pandas.DataFrame
            The peaks from suitable mass spectra.
        ret_times : numpy.ndarray
            The rentention times associated wit this peak.
        peak_bounds : Tuple, optional
            The minimum and maximum retention times for peak integration.

        Returns
        -------
        Feature
            The found feature.
        """
        tol = int(self._tol * feature_mz / 1e6)
        mz_lim = (feature_mz - tol, feature_mz + tol)
        mz_array = np.array([np.nan] * len(ret_times))
        int_array = np.zeros_like(ret_times)
        frag_peaks = peaks.loc[peaks["mz"].between(*mz_lim), :]
        if frag_peaks.empty:
            return None

        frag_peaks = frag_peaks.loc[
            utils.groupby_max(frag_peaks, "spectrum_idx", "intensity"), :
        ]

        if "index" not in frag_peaks.columns:
            within_rt = self._scans["scan_start_time"].between(
                ret_times.min(), ret_times.max()
            )
            scans = (
                self._scans.loc[within_rt, "spectrum_idx"]
                .reset_index(drop=True)
                .reset_index()
            )
            frag_peaks = frag_peaks.merge(scans, how="right")

        mz_array[frag_peaks["index"]] = frag_peaks["mz"]
        int_array[frag_peaks["index"]] = frag_peaks["intensity"]
        feat = Feature(
            query_mz=feature_mz,
            mean_mz=int(np.nanmean(mz_array)),
            row_ids=frag_peaks["rowid"].to_list(),
            ret_times=ret_times.copy(),
            intensities=np.nan_to_num(int_array),
        )
        if peak_bounds is not None:
            feat.lower_bound, feat.upper_bound = peak_bounds

        return feat

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
        top_feat = (
            self._peaks.iloc[[0], :].merge(self._scans, how="left").iloc[0, :]
        )
        min_rt = top_feat["scan_start_time"] - self._width
        max_rt = top_feat["scan_start_time"] + self._width
        within_rt = self._scans["scan_start_time"].between(min_rt, max_rt)
        ret_times = self._scans.loc[within_rt, "scan_start_time"].values
        feature = self._find_feature(
            feature_mz=int(top_feat["mz"]),
            peaks=self._peaks,
            ret_times=ret_times,
        )
        feature.update_bounds()
        print(feature.area)
        return feature


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
