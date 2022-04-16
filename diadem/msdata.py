"""A SQLite3 databse for DIA mass spectrometry data"""
import pickle
import inspect
import sqlite3
import logging
from pathlib import Path

from tqdm.auto import tqdm
from pyteomics.mzml import MzML

from . import utils
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
