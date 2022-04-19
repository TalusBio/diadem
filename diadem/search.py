"""A class to score peptides"""
import numpy as np
from abc import ABC, abstractmethod

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
    """

    def __init__(self, run, peptides, window_mz, width, tol):
        """Initialize a BaseWindowScorer"""
        self._run = run
        self._peptides = peptides
        self._window_mz = window_mz
        self._width = width
        self._tol = tol

    def _precursors(self):
        pass

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
        with self._run:
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
                return None, None, None

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
                ORDER BY fi.intensity DESC
                GROUP BY fi.mz;
                """,
                (min_mz, max_mz, min_rt, max_rt),
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
