"""Amino acid masses and other useful mass spectrometry calculations"""
import re
import numpy as np
import numba as nb
from numba.typed import Dict

# Constants
H = 1.007825035
O = 15.99491463
OH = O + H
H2O = 2 * H + O
PROTON = 1.00727646688
C13 = 1.003355


class PeptideMasses:
    """Calculate the precursor and fragment masses of peptides.

    Parameters
    ----------
    masses: dict of str, float (optional)
        Update the amino acid mass dictionary with new masses. This is
        how we can specify "static modifications".
    precision: int (optional)
        Calculated masses will be integerized to this many decimal places.

    Attributes
    ----------
    masses : dict of str, float
        The masses of amino acid residues.
    """

    _masses = {
        "n": 0.0,
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C": 103.009184505,  # + 57.02146,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        "U": 150.953633405,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        "O": 237.147726925,
        "c": 0.0,
    }

    def __init__(self, masses=None, precision=5):
        """Initialize the PeptideMasses object"""
        self.masses = Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float64,
        )

        self.masses.update(self._masses)
        self.precision = precision
        if masses is not None:
            self.masses.update(masses)

    def __len__(self):
        """The number of defined amino acids."""
        return len(self.masses)

    def to_float(self, mz_int):
        """Convert an integerized m/z back to a floating point representation.

        Parameters
        ----------
        mz_int : int
            The integerized m/z

        Returns
        -------
        float
            The canonical m/z value.
        """
        return _int2mz(mz_int, self.precision)

    def precursor(self, seq, mods=None, charge=None, n_isotopes=3):
        """Calculate the precursor mass of a peptide sequence.

        Parameters
        ----------
        seq : str
            The peptide sequence, without modifications.
        mods : list of float, optional
            Modification masses to consider at each position. The lengths of
            mods should be the length of seq plus two to account for N- and
            C-terminal modifications.
        charge : int, optional
            The charge state to consider. Use 'None' to get the neutral mass.
        n_isotopes : int, optional
            The number of C13 isotopes to return, starting from the
            monoisotopic mass.

        Yields
        ------
        float
            The precurosr monoisotopic m/z and requested C13 isotopes.
        """
        args = (seq, mods, charge, n_isotopes, self.precision, self.masses)
        for prec in _calc_precursor_mass(*args):
            yield prec

    def fragments(self, seq, mods=None, charge=2):
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
        float
            The m/z of the predicted b and y ions.
        """
        args = (seq, mods, charge, self.precision, self.masses)
        for frag in _calc_fragment_masses(*args):
            yield frag


############################################################
# Fast JIT-compiled functions to do the heavy lifting here #
############################################################
@nb.njit
def _seq2mass(seq, mods, vocab):
    """Convert a peptide sequence into an array of masses.

    Parameters
    ----------
    seq : str
        The peptide sequence.
    mods : np.ndarray
        The modifications at each position.
    vocab : numba.typed.Dict
        The amino acid vocabulary.

    Returns
    -------
    np.ndarray
        The mass at each position.
    """
    if seq.startswith("n"):
        seq = seq[1:]

    if seq.endswith("c"):
        seq = seq[:-1]

    out = np.empty(len(seq) + 2)
    if mods is None:
        mods = np.zeros(len(seq) + 2)

    if len(mods) != (len(seq) + 2):
        raise ValueError("'len(mods)' must equal 'len(seq) + 2'")

    for idx, (aa, mod) in enumerate(zip("n" + seq + "c", mods)):
        out[idx] = vocab[aa] + mod

    return out


@nb.njit
def _calc_precursor_mass(seq, mods, charge, n_isotopes, precision, vocab):
    """Calculate the precursor mass of a peptide sequence

    Parameters
    ----------
    seq : np.array
        The mass at each position.
    charge : int
        The charge state to consider. Use 'None' to get the neutral mass.
    n_isotopes : int
        The number of C13 isotopes to return, starting from the
        monoisotopic mass.
    precision : int
        How many decimal places to retain.

    Yields
    ------
    float
        The precurosr monoisotopic m/z and requested C13 isotopes.
    """
    mass = _seq2mass(seq, mods, vocab).sum() + H2O
    for isotope in range(n_isotopes):
        if isotope:
            mass += C13

        if charge is not None:
            mass = _mass2mz(mass, charge)

        yield _mz2int(mass, precision)


@nb.njit
def _calc_fragment_masses(seq, mods, charge, precision, vocab):
    """Calculate the b and y ions for a peptide sequence.

    Parameters
    ----------
    seq : np.array
        The mass at each position.
    charge : int, optional
        The precursor charge state to consider. If 1, only +1 fragment ions
        are returned. Otherwise, +2 fragment ions are returned.
    masses : nb.typed.Dict
        The mass dictionary to use.
    precision : int
        How many decimal places to retain.

    Yields
    ------
    float
        The m/z of the predicted b and y ions.
    """
    seq = _seq2mass(seq, mods, vocab)
    max_charge = min(charge, 2)
    for idx in range(2, len(seq)):
        b_mass = seq[:idx].sum()
        y_mass = seq[idx:].sum() + H2O
        for cur_charge in range(1, max_charge + 1):
            for frag in [b_mass, y_mass]:
                mz = _mass2mz(frag, cur_charge)
                yield _mz2int(mz, precision)


@nb.njit
def _mass2mz(mass, charge):
    """Calculate the m/z

    Parameters
    ----------
    mass : float
        The neutral mass.
    charge : int
        The charge.

    Returns
    -------
    float
       The m/z
    """
    return (mass / charge) + PROTON


@nb.njit
def _mz2int(moverz, precision):
    """Convert an m/z to an int

    Parameters
    ----------
    moverz : float
        The m/z value to convert.
    precision : int
        How many decimal places to retain.

    Returns
    -------
    int
        The intergerized m/z value.
    """
    return int(moverz * 10**precision)


@nb.njit
def _int2mz(mz_int, precision):
    """Convert an integer to the m/z.

    Parameters
    ----------
    mz_int : int
        The integerized m/z value.
    precision : int
        How many decimal places were retained.

    Returns
    -------
    float
        The m/z value.
    """
    return mz_int / 10**precision
