"""A base class for SQLite3 databases used by diadem"""
import pickle
import inspect
import sqlite3
from pathlib import Path
from abc import ABC, abstractmethod


class Database(ABC):
    """A base calsee for databases in diadem.

    Parameters
    ----------
    db_file : str
        The database file.
    force_ : bool
        Overwrite `db_file` if it already exists?
    """

    def __init__(self, db_file, force_):
        """Initialize the DB"""
        self._con = None
        self._cur = None
        self._force = bool(force_)
        self._db_file = Path(db_file)

        # Get the __init__ of the child instantiation:
        frame = inspect.currentframe().f_back
        *_, self._params = inspect.getargvalues(frame)
        for key in ["force_", "__class__", "self"]:
            del self._params[key]

        self._built = self._initialize()

    def __enter__(self):
        """Connect to the database"""
        self.connect()
        return self

    def __exit__(self, *args):
        """Close the database connection"""
        self.close()

    def connect(self):
        """Connect to the database."""
        self._con = sqlite3.connect(self._db_file)
        self._cur = self._con.cursor()

    def close(self):
        """Close the database connection"""
        self._con.close()
        self._con = None
        self._cur = None

    @property
    def params(self):
        """A dict of parameters used to initialize the PeptideDB"""
        return self._params

    @property
    def cur(self):
        """The database cursor"""
        return self._cur

    @property
    def con(self):
        """The database connection"""
        return self._con

    def _initialize(self):
        """Create a database file, only if needed.

        Check if the file already exists and if it does, verify that it has
        matching parameters. Otherwise, raise an error if force_ is not used.

        Returns
        -------
        bool
            True if the file already exists and False otherwise.
        """
        if self._db_file.exists() and not self._force:
            try:
                db_params = self.load_params(self._db_file)
                assert self._params == db_params
                return True

            except AssertionError:
                raise FileExistsError(
                    "The database file already exists and "
                    "the parameters do not match."
                )

        self._db_file.unlink(missing_ok=True)

        # Create the database:
        with self:
            self.cur.execute("PRAGMA foreign_keys=ON")
            self.cur.execute("CREATE TABLE parameters (params BLOB);")
            pkl_params = pickle.dumps(self._params)
            self.cur.execute(
                """
                INSERT INTO parameters (params)
                VALUES (?);
                """,
                (pkl_params,),
            )

            self._create_tables()
            self.con.commit()
            return False

    @staticmethod
    def load_params(db_file):
        """Load the parameters used to generate a PeptideDB.

        Parameters
        ----------
        db_file : str of Path
            The database file to load.

        Returns
        -------
        dict
            A dictionary of the parameters used to initialize the PeptideDB.
        """
        with sqlite3.connect(db_file) as con:
            cur = con.cursor()
            params = cur.execute("SELECT params FROM parameters").fetchone()

        return pickle.loads(params[0])

    @classmethod
    def from_file(cls, db_file):
        """Load previously created PeptideDB.

        Parameters
        ----------
        db_file : str or Path
            The database file to load.

        Returns
        -------
        PeptideDB
            The loaded PeptideDB.
        """
        params = cls.load_params(db_file)
        return cls(force_=False, **params)

    @abstractmethod
    def _create_tables(self):
        """Create database tables."""
        pass
