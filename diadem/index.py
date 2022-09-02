"""An inverted index mapping fragment ions to peptides and vice versa"""
import pickle
import logging
import itertools
import functools
from pathlib import Path
from collections import defaultdict

import mokapot
import numpy as np
import numba as nb
from tqdm import tqdm
from pyteomics.fasta import FASTA

from . import utils
from .masses import PeptideMasses

LOGGER = logging.getLogger(__name__)


class FragmentIndex:
    """The database to search against.

    Create a mapping between fragment ions and their generating peptides.

    fasta_files : str or Path or list of str or Path
        One or more FASTA files containing the proteins to digest.
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
    clip_nterm_methionine : bool, optional
        Temove methionine residues that occur at the protein N-terminus.
        Setting to `True` retains the original peptide and adds the clipped
        peptide to the database.
    charge_states : list of int, optional
        The charge states to consider.
    semi : bool, optional
        Require only on enzymatic terminus. Note that the protein database may
        become very large if set to True.
    decoy_prefix : str, optional
        The prefix prepended to decoy protein names.
    rng : int or numpy.random.Generator, optional
        The seed or generator to use for decoy generation.

    Attributes
    ----------
    proteins : dict of str, str
        A dictionary mapping protein accessions to protein sequences.
    peptides : dict of str, set of str
        A dictionary mapping peptide sequences to protein groups.
    modified_peptides : list of tuple(str, np.ndarray, bool)
        A list of modified peptides specified by the target peptide sequence,
        the packed modification array, and whether it is a target.
    precursors : list of tuple(int, int)
        A list of precursors specified by the index of the modified peptide and
        the charge state.
    frag2prec : dict of int, list of int
        A dictionary mapping integerized fragment m/z to the index of precursors
        that may have generated it.
    """

    _peptide = PeptideMasses()
    _init_static_mods = {57.02146: "C"}
    _init_variable_mods = {15.9949: ["M"]}

    def __init__(
        self,
        fasta_files,
        static_mods=None,
        variable_mods=None,
        max_mods=3,
        enzyme="[KR]",
        missed_cleavages=1,
        min_length=6,
        max_length=50,
        clip_nterm_methionine=True,
        charge_states=(2, 3),
        semi=False,
        decoy_prefix="decoy_",
        rng=None,
    ):
        """Initialize a PeptideIndex"""
        # Set simple private attributes:
        self._fasta_files = utils.listify(fasta_files)
        self._max_mods = max_mods
        self._enzyme = enzyme
        self._missed_cleavages = missed_cleavages
        self._min_length = min_length
        self._max_length = max_length
        self._clip_nterm_methionine = clip_nterm_methionine
        self._charge_states = utils.listify(charge_states)
        self._semi = bool(semi)
        self._rng = np.random.default_rng(rng)
        self._decoy_prefix = decoy_prefix
        self._mod_cols = None

        # The mappings:
        self.proteins = None
        self.peptides = None
        self.modified_peptides = None
        self.precursors = None
        self.frag2prec = None

        # Set static modifications
        self._static_mods = dict(self._init_static_mods)
        if static_mods is not None:
            self._static_mods.update(static_mods)
            for mod, residues in self._static_mods.items():
                for aa in utils.listify(residues):
                    self._peptide.masses[aa] += mod

        # Set variable modifications:
        if variable_mods is None:
            variable_mods = self._init_variable_mods

        self._variable_mods = variable_mods

        # Create a dictionary for modifying peptides:
        inverted_mods = defaultdict(list)
        for mod, residues in self._variable_mods.items():
            for residue in utils.listify(residues):
                inverted_mods[residue].append(mod)

        self._mod_order = []
        self._mod_vals = defaultdict(list)
        for aa in self._peptide.masses:
            masses = [0]
            masses += list(sorted(list(inverted_mods.get(aa, []))))
            for mass in masses:
                self._mod_vals[aa].append(mass)
                self._mod_order.append(mass)

        self._mod_order = np.array(self._mod_order)[:, None]

        # Permutations for decoys
        # Shuffle the non-terminal residues.
        self._perms = {
            i: [0] + list(self._rng.permutation(np.arange(1, i - 1))) + [-1]
            for i in range(self._min_length, self._max_length + 1)
        }

    @staticmethod
    def load(path):
        """Load a previous created FragmentIndex"""
        with Path(path).open("rb") as indata:
            return pickle.load(indata)

    def save(self, path="diadem.index.pkl"):
        """Save the FragmentIndex.

        Parameters
        ----------
        path : str or pathlib.Path
            The output file path.

        Returns
        -------
        selfd
        """
        path = Path(path)
        with path.open("wb+") as out:
            pickle.dump(self, out)

        return self

    def build(self):
        """Build the index.

        Returns
        -------
        self
        """
        residues = set(self._peptide.masses.keys())
        self._read_fastas()

        self.modification_blobs = []
        self.modified_peptides = []
        self.precursors = []
        self.frag2prec = defaultdict(list)
        self.prec2frag = defaultdict(list)

        skipped = []
        missing_aas = set()
        LOGGER.info("Building fragment index...")
        for target, prots in tqdm(self.peptides.items(), unit="peptides"):
            if no_token := set(target) - residues:
                skipped.append((target, ", ".join(prots)))
                missing_aas |= no_token

            # Create a decoy peptide and see if it exists already:
            decoy = "".join(target[i] for i in self._perms[len(target)])
            seqs = [target]
            if decoy not in self.peptides:
                seqs.append(decoy)

            # The second sequence is always the decoy.
            for is_decoy, seq in enumerate(seqs):
                for mods in self._modify(seq):
                    mod_pep_idx = len(self.modified_peptides)
                    packed_mods = self._pack_mods(mods)
                    self.modified_peptides.append(
                        (seq, packed_mods, not is_decoy)
                    )

                    for charge in self._charge_states:
                        prec_idx = len(self.precursors)
                        self.precursors.append((mod_pep_idx, charge))
                        frags = self.fragments(seq, mods, charge)
                        for frag in frags:
                            self.frag2prec[frag].append(prec_idx)

        return self

    def fragments(self, seq, mods, charge):
        """Calculate the b and y ions for a peptide sequence.

        Parameters
        ----------
        seq : str
            The peptide sequence, without modifications.
        mods : list of float, optional
            Modification masses to consider at each position. The lengths of
            mods should be the length of seq plus two to account for N- and
            C-terminal modifications.
        charge : int, optional
            The precursor charge state to consider. If 1, only +1 fragment ions
            are returned. Otherwise, +2 fragment ions are returned.

        Yields
        ------
        int
            The integerized m/z of the predicted b and y ions.
        """
        for frag in self._peptide.fragments(seq, mods, charge):
            yield frag

    def precursors_from_fragment(self, frag):
        """Look-up the precurors that may have generated a fragment ion.

        Parameters
        ----------
        frag : int
            The integerized fragment m/z.

        Yields
        ------
        seq : str
            The peptide sequence.
        mods : np.ndarray
            The modification masses at each position in the sequence.
        charge : int
            The charge of the precursor.
        """
        for prec_idx in self.frag2prec[frag]:
            yield self[prec_idx]

    def __getitem__(self, idx):
        """Retrieve a precursor by its index.

        Returns
        -------
        seq : str
            The peptide sequence.
        mods : np.ndarray
            The modification masses at each position in the sequence.
        charge : int
            The charge of the precursor.
        is_decoy : bool
            Is the peptide a decoy?
        """
        mod_pep_idx, charge = self.precursors[idx]
        seq, packed_mods, is_decoy = self.modified_peptides[mod_pep_idx]
        mods = self._unpack_mods(packed_mods, len(seq))
        return seq, mods, charge, is_decoy

    def _read_fastas(self):
        """Read the fasta files, digesting the protein sequence."""
        peptides = {}
        sequences = {}
        for fasta_file in self._fasta_files:
            LOGGER.info("Reading '%s'...", fasta_file)
            with FASTA(str(fasta_file)) as fas:
                for protein, seq in fas:
                    protein = protein.split(" ")[0]
                    seq = seq.upper()
                    peps = mokapot.digest(
                        seq,
                        enzyme_regex=self._enzyme,
                        missed_cleavages=self._missed_cleavages,
                        clip_nterm_methionine=self._clip_nterm_methionine,
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
            "- Found %i protein groups with %i peptide sequences.",
            len(proteins),
            len(peptides),
        )
        self.peptides = peptides
        self.proteins = proteins

    def _modify(self, seq):
        """Get all modified forms of a peptide.

        Yields
        ------
        mod_array : str
            The modifications
        """
        seq = "n" + seq + "c"
        mod_lists = [self._mod_vals[aa] for aa in seq]
        for mod_state in itertools.product(*mod_lists):
            mod_array = np.array(mod_state)
            n_mods = np.sum(mod_array > 0)
            if self._max_mods is not None and self._max_mods < n_mods:
                continue

            yield mod_array.astype(float)

    def _pack_mods(self, mod_array):
        """Pack an array of modification masses into a compact numpy aray.

        Parameters
        ----------
        mod_array : np.ndarray
            The numpy array containing the modification masses.

        Returns
        -------
        np.ndarray
            An integer numpy array representing the modifications as packed
            bits.
        """
        return np.packbits(mod_array[None, :] == self._mod_order)

    def _unpack_mods(self, packed_array, pep_length):
        """Unpack an array of modification masses.

        Parameters
        ----------
        packed_array : np.ndarray
            The bit representations of the modifications.
        pep_length : int
            The length of the peptide
        """
        unpacked = np.unpackbits(
            packed_array, count=(pep_length + 2) * len(self._mod_order)
        ).reshape((len(self._mod_order), -1))
        return _onehot2mass(unpacked, self._mod_order)


@nb.njit
def _onehot2mass(onehot_array, mods):
    """Get masses from the one-hot encoded modification array

    Parameters
    ----------
    onehot_array : np.array
        The one-hot encoded mass array.
    mods : np.array
        The modification mass array.

    Returns
    -------
    np.ndarray
        A 1D numpy array containing the mofidication mass at each position.
    """
    out = np.zeros(onehot_array.shape[1], dtype=np.float64)
    for row, mod in zip(onehot_array, mods[:, 0]):
        out[np.where(row)] = mod

    return out


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
