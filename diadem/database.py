"""The peptide database"""
import pickle
import inspect
import sqlite3
import logging
from pathlib import Path

import mokapot
import numpy as np
from tqdm.auto import tqdm

from . import utils
from .masses import PeptideMasses
from pyteomics.fasta import FASTA


class PeptideDB:
    """The peptide database to search against.

    Create a SQLite3 database containing the peptides and their fragment ions.
    This database is structured as an inverted index, allowing us to quickly
    look up the peptides that may have generated a fragment ion.

    Parameters
    ----------
    fasta_files : str or Path or list of str or Path
        One or more FASTA files containing the proteins to digest.
    db_file : str or Path
        The path of the created database file. The default is to use the stem of
        the first FASTA file, replacing the extension with ".db".
    static_mods : dict of str, float, optional
        Specify static modifications of the amino acids. Carbamidomethyl
        cysteine (+57) is used by default and must be explicitly changed. Use
        "n" and "c" to denote the c-terminus and n-terminus, respectively.
    variable_mods : dict of str, float or dict of str, array of float, optional
        Specify variable modifications of the amino acids. By default, only
        oxidized methionine (+16) is considered, but specifying other
        methionine modifications will overwrite it.
    max_mods : int, optional
        The maximum number of modifications per peptide.
    enzyme : str, optional
        A regular expression defining the enzyme specificity. The cleavage is
        interpreted as the end of the match. The default is trypsin, without
        proline suppression: "[KR]".
    missed_cleavages : int, optional
        The allowed number of missed cleavage sites.
    min_length : int, optional
        The minimum peptide length to consider.
    max_length : int, optional
        The maximum peptide length to consider.
    charge_states : list of int, optional
        The charge states to consider.
    semi : bool, optional
        Require only on enzymatic terminus. Note that the protein database may
        become very large if set to True.
    rng : int or numpy.random.Generator, optional
        The seed or generator to use for decoy generation.
    force_ : bool, optional
        Overwrite `db_file` if it already exists?
    """

    _pepcalc = PeptideMasses()
    _static_mods = {"C": 57.02146}
    _variable_mods = {"M": [15.9949]}
    _from_file = False

    def __init__(
        self,
        fasta_files,
        db_file=None,
        static_mods=None,
        variable_mods=None,
        max_mods=3,
        enzyme="[KR]",
        missed_cleavages=2,
        min_length=6,
        max_length=50,
        charge_states=(2, 3),
        semi=False,
        rng=None,
        force_=False,
    ):
        """Initialize a PeptideDB."""
        self._con = None  # The database connection
        self._cur = None  # The database cursor.
        self._built = False  # Indicate that the DB needs to be built.

        # Save the initialization parameters:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self._params = {k: v for k, v in zip(args, values)}
        del self._params["force_"]  # We don't want to keep this one.

        # Set simple private attributes:
        self._fasta_files = utils.listify(fasta_files)
        self._max_mods = max_mods
        self._enzyme = enzyme
        self._missed_cleavages = missed_cleavages
        self._min_length = min_length
        self._max_length = max_length
        self._charge_states = utils.listify(charge_states)
        self._semi = bool(semi)
        self._force = bool(force_)
        self._rng = np.random.default_rng(rng)

        # Set static modifications
        if static_mods is not None:
            self._static_mods.update(static_mods)
            for aa, mod in self._static_mods:
                self._pepcalc.masses[aa] += mod

        # Set variable modifications:
        if variable_mods is not None:
            self._variable_mods = {
                k: utils.listify(v) for k, v in variable_mods.items()
            }

        # Prepare the database file:
        if db_file is None:
            self._db_file = Path(Path(self._fasta_files[0]).stem + ".db")
        else:
            self._db_file = Path(db_file)

        self._init_db()
        self._build_db()

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

    def load_params(self):
        """Load initialization parameters from the database."""
        return self.cur.execute("SELECT params FROM parameters")[0]

    @property
    def cur(self):
        """The database cursor"""
        return self._cur

    @property
    def con(self):
        """The database connection"""
        return self._con

    def _init_db(self):
        """Create a database file, only if needed.

        Check if the file already exists and if it does, verify that it has
        matching parameters. Otherwise, raise an error if force_ is not used.
        """
        if self._db_file.exists() and not self._force:
            try:
                with self as db_conn:
                    db_params = db_conn.load_params()

                assert self._params == db_params
                return

            except AssertionError:
                raise FileExistsError(
                    "The database file already exists and "
                    "the parameters do not match."
                )

        self._db_file.unlink(missing_ok=True)

        # Create the database:
        with self as db_conn:
            self.cur.execute("CREATE TABLE parameters (params BLOB)")
            pkl_params = pickle.dumps(self._params)
            self.cur.execute("INSERT INTO parameters (?)", pkl_params)
            self.cur.execute("CREATE TABLE fragments (mz INT, precursor INT)")
            self.cur.execute(
                """
                CREATE TABLE precursors (
                    precursor INT,
                    sequence STRING,
                    charge INT,
                    mz INT,
                    decoy BOOL
                )
                """
            )
            self.cur.commit()

    def _digest_proteins(self):
        """Digest proteins into peptides.

        Returns
        -------
        set of str
            The unique peptides.
        """
        peptides = {}
        for fasta_file in self._fasta_files:
            with FASTA(str(fasta_file)) as fas:
                for seq in fas:
                    peptides |= mokapot.digest(
                        seq,
                        self._enzyme,
                        self._missed_cleavages,
                        True,
                        self._min_length,
                        self._max_length,
                        self._semi,
                    )

        return peptides

    def _add_peptide(self, seq):
        """Fragment a peptide and add it to the database.

        Parameter
        ---------
        seq : str
            The peptide sequence
        """
        for charge in self._charge_states:
            pass


def _modify_peptide(seq, mods, max_mods):
    """Get all modification states of a peptide"""
    pass
