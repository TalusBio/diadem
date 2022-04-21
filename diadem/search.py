"""A class to score peptides"""
import itertools
from abc import ABC, abstractmethod

import numpy as np

from . import feature, utils


class PeptideSearcher:
    """Score peptides against features in a the DIA mass spectrometry data."""

    pass


class BaseWindowScorer(ABC):
    """Score peptides against features in the DIA mass spectrometry data.

    Child classes implement specific score functions.

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
    """

    def __init__(self, run, peptides, window_mz, width, tol, open_, min_corr):
        """Initialize a BaseWindowScorer"""
        self._run = run
        self._peptides = peptides
        self._window_mz = window_mz
        self._width = width
        self._tol = tol
        self._open = bool(open_)
        self._min_corr = min_corr

    def __iter__(self):
        """Iterate through the features in a run"""
        with self._run:
            while True:
                peak = self._next_peak()
                if peak[0] is None:
                    break

                yield peak

    @abstractmethod
    def score(self, areas):
        """Score each precursor using a score function.

        Pareameters
        -----------
        areas : list of numpy.ndarray
            The intensities of the detected fragment ions. Each element in the
            list corresponds to one precursor. Any undetected fragment ion will
            be 0.

        Returns
        -------
        list of float
            The score for each precursor, where higher is bettter.
        """
        pass

    def search(self):
        """Perform the database search"""
        for feat_mz, area, bounds, spectra in self:
            precursors, fragments = self._lookup_precursors(feat_mz)
            peak_areas = self._find_corr_peaks(fragments, bounds, spectra)

    def _find_corr_peaks(self, fragment_list, area, bounds, spectra):
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
        list of numpy.ndarray
            The peak areas for all of the theoretical peaks of a precursor.
        """
        sizes = [len(l) for l in fragment_list]
        fragment_list = itertools.chain.from_iterable(fragment_list)

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

        fragments = self._peptides.precursors_to_fragments(
            precursors,
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
            SELECT TOP(1) fi.mz, fr.scan_start_time
                FROM fragment_ions AS fi
            LEFT JOIN fragments AS fr
                ON fi.spectrum_idx=fr.spectrum_idx
            LEFT JOIN used_ions AS ui
                ON ui.fragment_idx=fi.ROWID
            WHERE f.isolation_window_lower <= ?
                AND f.isolation_window_upper >= ?
                AND ui.fragment_idx IS NULL
            ORDER BY fi.intensity DESC;
            """,
            self._window_mz,
        )

        frag_mz, ret_time = top_query.fetch()
        if not frag_mz:
            return None, None, None, None

        ppm = int(self._tol * frag_mz / 1e6)
        min_mz = frag_mz - ppm
        max_mz = frag_mz + ppm
        min_rt = ret_time - self._width
        max_rt = ret_time + self._width

        feat_query = self._run.cur.execute(
            """
            SELECT TOP(1) fi.ROWID, fi.*, fr.scan_start_time
                FROM fragment_ions AS fi
            LEFT JOIN fragments AS fr
                ON fi.spectrum_idx=fr.spectrum_idx
            WHERE fi.mz BETWEEN ? AND ?
                AND fr.scan_start_time BETWEEN ? AND ?
                AND f.isolation_window_lower <= ?
                AND f.isolation_window_upper >= ?
            ORDER BY fr.spectrum_idx, fi.intensity DESC
            GROUP BY fr.spectrum_idx;
            """,
            (min_mz, max_mz, min_rt, max_rt, *self._window_mz),
        )

        row_id, mz_array, int_array, spec_idx, ret_times = list(
            zip(*feat_query.fetchall())
        )

        bounds = feature.build(int_array)
        peak_area = feature.integrate(int_array, ret_times, bounds)
        mean_mz = int(np.mean(mz_array[bounds[0] : bounds[1]]))
        spectrum_bounds = (spec_idx[0], spec_idx[1])
        self._run.remove_ions(row_id)

        return mean_mz, peak_area, bounds, spectrum_bounds
