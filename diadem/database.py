"""The peptide database"""
import pickle
import inspect
import sqlite3
import logging
from pathlib import Path
from collections import defaultdict

import mokapot
import numpy as np
from tqdm.auto import tqdm

from . import utils
from .masses import PeptideMasses
from pyteomics.fasta import FASTA

LOGGER = logging.getLogger(__name__)


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

        # Permutations for decoys:
        self._perms = {
            i: self._rng.shuffle(np.arange(i - 2))
            for i in range(self._min_length, self._max_length)
        }

        self._init_db()
        with self:
            self._build_db()
            self.res = self.cur.execute(
                "SELECT COUNT(*) FROM peptides"
            ).fetchone()

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
        return self.cur.execute("SELECT params FROM parameters").fetchone()

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

            self.cur.execute(
                """
                CREATE TABLE proteins (
                    accession STRING UNIQUE,
                    sequence STRING
                );
                """
            )

            self.cur.execute(
                """
                CREATE TABLE peptides (
                    peptide_idx INT,
                    protein_idx INT REFERENCES proteins(ROWID),
                    start INT,
                    end INT
                );
                """
            )

            self.cur.execute(
                """
                CREATE TABLE modpeptides (
                    peptide_idx INT REFERENCES peptides(peptide_idx)
                );
                """
            )

            self.cur.execute(
                """
                CREATE TABLE precursors (
                    modpeptide_idx INT REFERENCES modpeptides(ROWID),
                    charge INT,
                    mz INT
                );
                """
            )

            self.cur.execute(
                """
                CREATE TABLE fragments (
                    mz INT,
                    precursor_idx INT REFERENCES precursors(ROWID)
                );
                """
            )

            for aa, masses in self._variable_mods.items():
                for mass in masses:
                    mod = f"{aa}[{mass:+.4f}]".replace('"', '""')
                    self.cur.execute(
                        f"""
                        ALTER TABLE modpeptides
                        ADD COLUMN "{mod}" BLOB;
                        """
                    )

            self.con.commit()

    def _build_db(self):
        """Build the database."""
        peptides, proteins = self._read_fastas()
        peptide_idx = 1
        desc = "Building database"
        for seq, prots in tqdm(peptides.items(), desc=desc, unit="peptides"):
            prot_rows = [(p, proteins[p]) for p in prots]
            self.cur.executemany(
                """
                INSERT OR IGNORE INTO proteins(accession, sequence)
                VALUES (?, ?);
                """,
                prot_rows,
            )

            pep_rows = []
            for prot in prots:
                start = proteins[prot].find(seq)
                end = start + len(seq)
                pep_rows.append((peptide_idx, start, end, prot))

            self.cur.executemany(
                """
                INSERT OR IGNORE INTO peptides
                SELECT ?, proteins.ROWID, ?, ?
                FROM proteins
                WHERE accession = ?
                LIMIT 1;
                """,
                pep_rows,
            )
            peptide_idx += 1

            # for modseq in self._peptide_mods(seq):
            #    pass

            self.con.commit()

    def _peptide_mods(self, seq):
        """Get all modified forms of a peptide.

        Yields
        ------
        str
            The modified peptide sequence.
        """
        mod_pos = {}
        for mod in self._variable_mods.keys():
            pass

        for x in [seq]:
            yield x

    def _read_fastas(self):
        """Read the fasta files, digesting the protein sequence.

        Returns
        -------
        peptides : dict of str, set of str
            The proteins mapped to their peptides.
        proteins : dict of str, str
            The proteins mapped to their sequence.
        """
        peptides = {}
        sequences = {}
        for fasta_file in self._fasta_files:
            with FASTA(str(fasta_file)) as fas:
                desc = "Digesting proteins"
                for protein, seq in tqdm(fas, desc=desc, unit="proteins"):
                    protein = protein.split(" ")[0]
                    peps = mokapot.digest(
                        seq,
                        enzyme_regex=self._enzyme,
                        missed_cleavages=self._missed_cleavages,
                        clip_nterm_methionine=True,
                        min_length=self._min_length,
                        max_length=self._max_length,
                        semi=self._semi,
                    )

                    if peps:
                        sequences[protein] = seq
                        peptides[protein] = peps

        peptides = _group_proteins(peptides)
        proteins = {}
        for prot in set().union(*peptides.values()):
            proteins[prot] = sequences[prot.split(";")[0]]

        LOGGER.info(
            "Found %i protein groups with %i peptides.",
            len(proteins),
            len(peptides),
        )
        return peptides, proteins


def _group_proteins(proteins):
    """Group proteins with proper subsets.

    Parameters
    ----------
    proteins : dict of str, set of str
       A map of proteins to their peptides.

    Returns
    -------
    peptides : dict str, set of str
        The peptides mapped to their protein groups.
    """
    # This is very similar to the counterpart in mokapot.
    peptides = defaultdict(set)
    for prot, peps in proteins.items():
        for pep in peps:
            peptides[pep].add(prot)

    grouped = {}
    for prot, peps in sorted(proteins.items(), key=lambda x: -len(x[1])):
        if not grouped:
            grouped[prot] = peps
            continue

        matches = set.intersection(*[peptides[p] for p in peps])
        matches = [m for m in matches if m in grouped]

        # If the entry is unique:
        if not matches:
            grouped[prot] = peps
            continue

        # Create new entries from subsets:
        for match in matches:
            new_prot = ";".join([match, prot])
            grouped[new_prot] = grouped.pop(match)

            # Update peptides
            for pep in grouped[new_prot]:
                peptides[pep].remove(match)
                peptides[pep].add(new_prot)
                if prot in peptides[pep]:
                    peptides[pep].remove(prot)

    return peptides
