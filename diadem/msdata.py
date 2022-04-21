"""A SQLite3 databse for DIA mass spectrometry data"""
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML

from . import utils
from .feature import Feature
from .database import Database

LOGGER = logging.getLogger(__name__)


class DIARunDB(Database):
    """An SQLite3 database of DIA mass spectrometry data.

    Create an SQLite3 database containing the raw mass spectrometry data
    from an data-independent acquisition (DIA) experiment. These files
    contain a subset of the information normally found in an mzML file and
    are designed to quickly query peaks and their associated spectra.

    Parameters
    ----------
    ms_data_file : str or Path
        The mass spectrometry data file to parse. Currently only mzML files
        are supported.
    db_file : str or Path, optional
        The database file to write. By default this is `ms_data_file` suffixed
        with ".db".
    force_ : bool, optional
        Overwrite the database file if it already exists.
    """

    def __init__(self, ms_data_file, db_file=None, force_=False):
        """Initialize a DIARunDB"""
        self._ms_data_file = Path(ms_data_file)

        if db_file is None:
            db_file = self._ms_data_file.name + ".db"

        super().__init__(db_file, force_)
        if not self._built:
            with self:
                self._build_db()

    def _create_tables(self):
        """Create the database tables."""
        self.cur.execute(
            """
            CREATE TABLE precursors (
                spectrum_idx INT PRIMARY KEY,
                scan_start_time FLOAT,
                total_ion_current FLOAT
            );
            """
        )

        self.cur.execute(
            """
            CREATE TABLE fragments (
                spectrum_idx INT PRIMARY KEY,
                scan_start_time FLOAT,
                total_ion_current FLOAT,
                isolation_window_upper FLOAT,
                isolation_window_center FLOAT,
                isolation_window_lower FLOAT
            );
            """
        )

        self.cur.execute(
            """
            CREATE TABLE precursor_ions (
                mz INT,
                intensity REAL,
                spectrum_idx INT REFERENCES precursors(spectrum_idx)
            )
            """
        )

        self.cur.execute(
            """
            CREATE TABLE fragment_ions (
                mz INT,
                intensity REAL,
                spectrum_idx INT REFERENCES fragments(spectrum_idx)
            )
            """
        )

        self.cur.execute(
            """
            CREATE TABLE used_ions (
                fragment_idx INT REFERENCES fragments(ROWID)
            );
            """
        )

    def find_features(self, mz_values, window, ret_time=None, tol=10):
        """Find features in the DIA data.

        Parameters
        ----------
        mz_values : list of float
            The m/z values to find.
        window : tuple of int
            The precursor isolation window bounds. `None` will instead
            look for the precursor signals.
        ret_time : tuple of float
            The retention time window to look in. `None` will search the
            entire run.
        tol : float
            The matching tolerance in ppm.

        Returns
        -------
        list of Features
            The extracted features.
        """
        if ret_time is None:
            ret_time = (0.0, 1e6)

        indices, ret_times = self._rt2scans(*ret_time, window)
        min_idx = min(indices)

        features = []
        for mz_val in utils.listify(mz_values):
            if not isinstance(mz_val, int):
                mz_val = utils.mz2int(mz_val)

            tol_val = int(tol * mz_val / 1e6)
            query = self.cur.execute(
                """
                SELECT TOP(1) fi.ROWID, fi.mz, fi.intensity, fr.spectrum_idx
                    FROM fragment_ions AS fi
                LEFT JOIN fragments AS fr
                    ON fi.spectrum_idx=fr.spectrum_idx
                WHERE fi.mz BETWEEN ? AND ?
                    AND fr.scan_start_time BETWEEN ? AND ?
                    AND fr.isolation_window_lower <= ?
                    AND fr.isolation_window_upper >= ?
                ORDER BY fr.spectrum_idx, fi.intensity DESC
                GROUP BY fr.spectrum_idx;
                """,
                (mz_val - tol_val, mz_val + tol_val, *ret_time, *window),
            )

            if not query:
                features.append(None)
                continue

            mz_array = np.array([np.nan] * len(ret_times))
            row_array = np.array([np.nan] * len(ret_times))
            int_array = np.zeros_like(ret_times)
            for row_id, feat_mz_val, int_val, spec_idx in query:
                idx = spec_idx - min_idx
                mz_array[idx] = feat_mz_val
                row_array[idx] = row_id
                int_array[idx] = int_val

            mz_array = np.array(mz_array)
            feat = Feature(
                query_mz=mz_val,
                mean_mz=np.nanmean(mz_array),
                row_ids=row_array.tolist(),
                ret_times=ret_times.copy(),
                intensities=int_array,
            )

            features.append(feat)

        return features

    def reset(self):
        """Reset the used_ions table."""
        self.cur.execute("""DELETE FROM used_ions;""")
        self.con.commit()

    def remove_ions(self, frag_row):
        """Add multiple fragment ions to the used_ion table.

        Parameters
        ----------
        fragment_mzs : list of int
            The integerized m/z

        """
        self.cur.executemany(
            "INSERT INTO used_ions VALUES (?);",
            [(f,) for f in utils.listify(frag_row)],
        )
        self.con.commit()

    def _rt2scans(self, min_rt, max_rt, window=None):
        """Get the scans for a DIA window at a specific retention time.

        Parameters
        ----------
        min_rt : float
            The lower bound of the retention time window.
        max_rt : float
            The upper bound of the retention time window.
        window : tuple of float
            The upper and lower bound of the DIA window. `None` will specify
            precursors.

        Returns
        -------
        indices : list of int
            The scan indices for the scan of interest.
        ret_times : np.array
            The retention times
        """
        if window is None:
            table = "precursors"
        else:
            table = "fragments"

        indices = self.cur.execute(
            f"""
            SELECT UNIQUE spectrum_idx, scan_start_time FROM {table}
            WHERE isolation_window_lower >= ?
                AND isolation_window_upper <= ?
                AND scan_start_time BETWEEN ? AND ?
            ORDER BY spectrum_idx
            """,
            (*window, min_rt, max_rt),
        )

        indices, ret_times = list(zip(*indices.fetchall()))
        return indices, np.array(ret_times)

    def _build_db(self):
        """Build the database."""
        precursors_rows = []
        fragments_rows = []
        precursor_ions_rows = []
        fragment_ions_rows = []
        insert_counter = 0
        with MzML(str(self._ms_data_file)) as mzdata:
            desc = f"Converting {self._ms_data_file}"
            for spectrum in tqdm(mzdata, desc=desc, unit="spectra"):
                insert_counter += 1
                idx = spectrum["index"]
                start = spectrum["scanList"]["scan"][0]["scan start time"]
                tic = spectrum["total ion current"]
                is_fragment = spectrum["ms level"] != 1
                spec_arrays = (
                    spectrum["m/z array"].tolist(),
                    spectrum["intensity array"].tolist(),
                )
                ions = [
                    (utils.mz2int(m), i, idx) for m, i in zip(*spec_arrays)
                ]
                if is_fragment:
                    precursor = spectrum["precursorList"]["precursor"][0]
                    window = precursor["isolationWindow"]
                    center = window["isolation window target m/z"]
                    lower = center - window["isolation window lower offset"]
                    upper = center + window["isolation window upper offset"]
                    fragment_ions_rows += ions
                    fragments_rows.append(
                        (idx, start, tic, upper, center, lower)
                    )

                else:
                    precursor_ions_rows += ions
                    precursors_rows.append((idx, start, tic))

                if insert_counter >= 10000:
                    self._update_db(
                        precursors_rows,
                        fragments_rows,
                        precursor_ions_rows,
                        fragment_ions_rows,
                    )
                    insert_counter = 0
                    precursors_rows = []
                    fragments_rows = []
                    precursor_ions_rows = []
                    fragment_ions_rows = []

        self._update_db(
            precursors_rows,
            fragments_rows,
            precursor_ions_rows,
            fragment_ions_rows,
        )
        self.con.commit()

    def _update_db(
        self,
        precursors_rows,
        fragments_rows,
        precursor_ions_rows,
        fragment_ions_rows,
    ):
        """Update the database.

        Parameters
        ----------
        precursors_rows : list of tuples
            The rows to insert into the precursors table.
        fragments_rows : list of tuples
            The rows to insert into the fragments table.
        precursor_ions_rows : list of tuples
            The rows to insert into the precursor_ions table.
        fragment_ions_rows : list of tuples
            The rows to insert into the fragment_ions table.
        """
        self.cur.executemany(
            """
            INSERT OR IGNORE INTO precursors
            VALUES (?, ?, ?);
            """,
            precursors_rows,
        )

        self.cur.executemany(
            """
            INSERT OR IGNORE INTO fragments
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            fragments_rows,
        )

        self.cur.executemany(
            """
            INSERT OR IGNORE INTO precursor_ions
            VALUES (?, ?, ?)
            """,
            precursor_ions_rows,
        )

        self.cur.executemany(
            """
            INSERT OR IGNORE INTO fragment_ions
            VALUES (?, ?, ?)
            """,
            fragment_ions_rows,
        )
