"""The peptide database"""
import pickle
import inspect
import sqlite3
import logging
import itertools
from pathlib import Path
from collections import defaultdict

import mokapot
import numpy as np
from tqdm.auto import tqdm
from pyteomics.fasta import FASTA

from . import utils
from .masses import PeptideMasses

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
    decoy_prefix : str, optional
        The prefix prepended to decoy protein names.
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
        decoy_prefix="decoy_",
        rng=None,
        force_=False,
    ):
        """Initialize a PeptideDB."""
        self._con = None  # The database connection
        self._cur = None  # The database cursor.
        self._built = False  # Indicate that the DB needs to be built.

        # Save the initialization parameters:
        frame = inspect.currentframe()
        *_, self._params = inspect.getargvalues(frame)
        for key in ["force_", "frame", "self"]:
            del self._params[key]

        print(self._params)

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
        self._decoy_prefix = decoy_prefix
        self._mod_cols = None

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

        # Also create the modification strings:
        self._mod_strings = defaultdict(list)
        for aa in self._pepcalc.masses:
            masses = list(self._variable_mods.get(aa, []))
            masses.append(None)
            for mass in masses:
                if mass is not None:
                    self._mod_strings[aa].append(f"{aa}{mass:+.4f}")
                else:
                    self._mod_strings[aa].append(aa)

        # Prepare the database file:
        if db_file is None:
            self._db_file = Path(Path(self._fasta_files[0]).stem + ".db")
        else:
            self._db_file = Path(db_file)

        # Permutations for decoys:
        self._perms = {
            i: [0] + list(self._rng.permutation(np.arange(1, i - 1))) + [-1]
            for i in range(self._min_length, self._max_length + 1)
        }

        skip_build = self._init_db()
        if not skip_build:
            with self:
                self._build_db()

    def __enter__(self):
        """Connect to the database"""
        self.connect()
        return self

    def __exit__(self, *args):
        """Close the database connection"""
        self.close()

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

    def connect(self):
        """Connect to the database."""
        self._con = sqlite3.connect(self._db_file)
        self._cur = self._con.cursor()

    def close(self):
        """Close the database connection"""
        self._con.close()
        self._con = None
        self._cur = None

    def fragment_to_precursor(self, frag_mz, tol=10, prec_mz=None):
        """Find peptides matching a fragment mass

        Parameters
        ----------
        frag_mz : float or int
            The m/z of a fragment ion to look up.
        tol : float
            The tolerance to use in ppm.
        prec_mz : tuple of float
            The precursor m/z range to consider. None considers
            everything.

        Returns
        -------
        list of tuple int
            The indices of precursors with matching fragments.
            This is a list of tuples instead of a list of int
            so it can easily be used with the sqlite3
            `executemany()` method.
        """
        if not isinstance(frag_mz, int):
            frag_mz = utils.mz2int(frag_mz)

        ppm = int(tol * frag_mz / 1e6)
        min_mz = frag_mz - ppm
        max_mz = frag_mz + ppm

        # For open modifications searching:
        if prec_mz is None:
            ret = self.con.execute(
                """
                SELECT DISTINCT precursor_idx FROM fragments
                WHERE mz BETWEEN ? AND ?;
                """,
                (min_mz, max_mz),
            )
            return ret.fetchall()

        # For closed searching:
        prec_min_mz = utils.mz2int(prec_mz[0])
        prec_max_mz = utils.mz2int(prec_mz[1])
        ret = self.con.execute(
            """
            SELECT DISTINCT fragments.precursor_idx
            FROM fragments
            JOIN precursors
            ON fragments.precursor_idx=precursors.ROWID
            WHERE precursors.mz BETWEEN ? AND ?
            AND fragments.mz BETWEEN ? AND ?;
            """,
            (prec_min_mz, prec_max_mz, min_mz, max_mz),
        )

        return ret.fetchall()

    def precursor_to_fragments(self, precursor_idx, to_int=False):
        """Find the fragments for a precursor.

        Parameters
        ----------
        precursor_idx : int
            The index of the precursor.
        to_int : bool
            Report the m/z as a fixed precision integer?

        Returns
        -------
        list of int or float
            The fragment m/z.
        """
        ret = self.con.execute(
            """
            SELECT DISTINCT mz
            FROM fragments
            WHERE precursor_idx = ?;
            """,
            (precursor_idx,),
        )

        if not to_int:
            return [utils.int2mz(f[0]) for f in ret]

        return [f[0] for f in ret]

    def _init_db(self):
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
                    end INT,
                    target BOOL
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

            self._mod_cols = []
            for aa, masses in self._variable_mods.items():
                for mass in masses:
                    mod = f"{aa}[{mass:+.4f}]".replace('"', '""')
                    self._mod_cols.append(mod)
                    self.cur.execute(
                        f"""
                        ALTER TABLE modpeptides
                        ADD COLUMN "{mod}" BLOB;
                        """
                    )

            self.con.commit()
            return False

    def _build_db(self):
        """Build the database."""
        peptides, proteins = self._read_fastas()
        peptide_idx = 0
        modpeptide_idx = 0
        prec_idx = 0
        commit_counter = 0
        prot_rows = []
        pep_rows = []
        mod_rows = []
        prec_rows = []
        frag_rows = []
        desc = "Building database"
        unit = "peptides"
        for target, prots in tqdm(peptides.items(), desc=desc, unit=unit):
            # Proteins
            prot_rows += [(p, proteins[p]) for p in prots]
            prot_starts = [[proteins[p].find(target) for p in prots]]

            # Create a decoy peptide and see if it exists:
            decoy = self._generate_decoy(target)
            if decoy not in peptides:
                seqs = [target, decoy]
                is_target = [True, False]
                prot_starts *= 2
            else:
                seqs = [target]
                is_target = [True]

            for seq, tval, starts in zip(seqs, is_target, prot_starts):
                peptide_idx += 1  # Increment peptides
                for start, prot in zip(starts, prots):
                    end = start + len(seq)
                    pep_rows.append((peptide_idx, start, end, prot, tval))

                # Modified Peptides
                for modseq, modblobs in self._peptide_mods(seq):
                    modpeptide_idx += 1  # Increment modified peptides
                    mod_rows.append((peptide_idx, *modblobs))
                    # Precursors
                    for charge in self._charge_states:
                        prec_idx += 1  # increment precursors
                        prec_mz = next(
                            self._pepcalc.precursor(modseq, charge, 1)
                        )
                        prec_rows.append(
                            (modpeptide_idx, charge, utils.mz2int(prec_mz))
                        )
                        frag_rows += [
                            (utils.mz2int(f), prec_idx)
                            for f in self._pepcalc.fragments(modseq, charge)
                        ]

            commit_counter += 1
            if commit_counter >= 10000:
                self._update_db(
                    prot_rows,
                    pep_rows,
                    mod_rows,
                    prec_rows,
                    frag_rows,
                )
                commit_counter = 0
                prot_rows = []
                pep_rows = []
                mod_rows = []
                prec_rows = []
                frag_rows = []

        self._update_db(prot_rows, pep_rows, mod_rows, prec_rows, frag_rows)
        self.con.commit()

    def _update_db(self, prot_rows, pep_rows, mod_rows, prec_rows, frag_rows):
        """Update the db."""
        self.cur.executemany(
            """
            INSERT OR IGNORE INTO proteins(accession, sequence)
            VALUES (?, ?);
            """,
            prot_rows,
        )

        self.cur.executemany(
            """
            INSERT OR IGNORE INTO peptides
            SELECT ?, proteins.ROWID, ?, ?, ?
            FROM proteins
            WHERE accession = ?
            LIMIT 1;
            """,
            pep_rows,
        )

        self.cur.executemany(
            f"""
            INSERT OR IGNORE INTO modpeptides
            VALUES ({"?, "*len(self._mod_cols)}?)
            """,
            mod_rows,
        )

        self.cur.executemany(
            """
            INSERT OR IGNORE INTO precursors
            VALUES (?, ?, ?)
            """,
            prec_rows,
        )

        self.cur.executemany(
            """
            INSERT OR IGNORE INTO fragments
            VALUES (?, ?)
            """,
            frag_rows,
        )

    def _peptide_mods(self, seq):
        """Get all modified forms of a peptide.

        Yields
        ------
        str
            The modified peptide sequence.
        """
        mod_lists = [self._mod_strings[aa] for aa in seq]
        for mod_pep in itertools.product(*mod_lists):
            mod_blobs = []
            n_mods = 0
            for mod_col in self._mod_cols:
                bool_array = [aa == mod_col for aa in mod_pep]
                n_mods += sum(bool_array)
                if n_mods > self._max_mods:
                    continue

                blob = utils.bools2bytes(bool_array)
                mod_blobs.append(blob)

            yield "".join(mod_pep), mod_blobs

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
                        clip_nterm_methionine=False,
                        min_length=self._min_length,
                        max_length=self._max_length,
                        semi=self._semi,
                    )

                    if peps:
                        sequences[protein] = seq
                        peptides[protein] = peps

        peptides = group_proteins(peptides)
        proteins = {}
        for prot in set().union(*peptides.values()):
            proteins[prot] = sequences[prot.split(";")[0]]

        LOGGER.info(
            "Found %i protein groups with %i peptides.",
            len(proteins),
            len(peptides),
        )
        return peptides, proteins

    def _generate_decoy(self, seq):
        """Shuffle target peptides returning a set of decoys.

        Parameters
        ----------
        seq : str
            The target peptide sequence.

        Returns
        -------
        str
            The decoy peptide sequence.
        """
        return "".join(seq[i] for i in self._perms[len(seq)])


def group_proteins(proteins):
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
