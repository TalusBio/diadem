"""Amino acid masses and other useful mass spectrometry calculations"""
import re
import numpy as np
import numba as nb

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

    Attributes
    ----------
    masses : dict of str, float
        The masses of amino acid residues.
    hydrogen : float
        The mass of hydrogen.
    oxygen : float
        The mass of oxygen.
    oh : float
        The mass of OH.
    h2o : float
        The mass of water, H2O.
    proton : float
        The mass of a proton.
    c13_diff : float
        The mass difference between C13 from C12.
    """

    _seq_regex = r"([A-Z]|[+-][\d\.]*)"
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

    def __init__(self, masses=None):
        """Initialize the PeptideMasses object"""
        self.masses = dict(self._masses)  # prevent updating the class :P
        if masses is not None:
            self.masses.update(masses)

    def __len__(self):
        """The number of defined amino acids."""
        return len(self.masses)

    def precursor(self, seq, charge=None, n_isotopes=3):
        """Calculate the precursor mass of a peptide sequence.

        Parameters
        ----------
        seq : str
            The peptide sequence, with modifications. Modification can be
            denoted using a '+' or '-' followed by the modification mass.
            For example, 'DIADEM+16K' or 'DIADEM[+16]K' will work.
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
        seq = self.parse(seq)
        for prec in _calc_precursor_mass(seq, charge, n_isotopes):
            yield prec

    def fragments(self, seq, charge=2):
        """Calculate the b and y ions for a peptide sequence.

        Parameters
        ----------
        seq : str
            The peptide sequence, with modifications. Modification can be
            denoted using a '+' or '-' followed by the modification mass.
            For example, 'DIADEM+16K' or 'DIADEM[+16]K' will work.
        charge : int, optional
            The precursor charge state to consider. If 1, only +1 fragment ions
            are returned. Otherwise, +2 fragment ions are returned.

        Yields
        ------
        float
            The m/z of the predicted b and y ions.
        """
        for frag in _calc_fragment_masses(self.parse(seq), charge):
            yield frag

    def parse(self, seq):
        """Parse a string into a list of masses

        Parameters
        ----------
        seq : str
            The peptide sequence, with modifications. Modification can be
            denoted using a '+' or '-' followed by the modification mass.
            For example, 'DIADEM+16K' or 'DIADEM[+16]K' will work.

        Returns
        -------
        list of float
            The amino acid masses, including modifications.
        """
        out = []
        for aa in re.findall(self._seq_regex, seq):
            if aa not in self.masses:
                out[-1] += float(aa)
            else:
                out.append(self.masses[aa])

        return np.array(out)


@nb.njit
def _calc_precursor_mass(seq, charge, n_isotopes):
    """Calculate the precursor mass of a peptide sequence

    Parameters
    ----------
    seq : list of str and float
        The peptide sequence, with modifications. Modification can be
        denoted using a '+' or '-' followed by the modification mass.
        For example, 'DIADEM+16K' or 'DIADEM[+16]K' will work.
    charge : int
        The charge state to consider. Use 'None' to get the neutral mass.
    n_isotopes : int
        The number of C13 isotopes to return, starting from the
        monoisotopic mass.

    Yields
    ------
    float
        The precurosr monoisotopic m/z and requested C13 isotopes.
    """
    mass = sum(seq) + H2O
    for isotope in range(n_isotopes):
        if isotope:
            mass += C13

        if charge is not None:
            yield _mass2mz(mass, charge)
        else:
            yield mass


@nb.njit
def _calc_fragment_masses(seq, charge):
    """Calculate the b and y ions for a peptide sequence.

    Parameters
    ----------
    seq : list of str
        The peptide sequence, with modifications.
    charge : int, optional
        The precursor charge state to consider. If 1, only +1 fragment ions
        are returned. Otherwise, +2 fragment ions are returned.
    masses : nb.typed.Dict
        The mass dictionary to use.

    Yields
    ------
    float
        The m/z of the predicted b and y ions.
    """
    max_charge = min(charge, 2)
    for idx in range(1, len(seq) - 1):
        b_mass = sum(seq[:idx])
        y_mass = sum(seq[idx:]) + H2O
        for cur_charge in range(1, max_charge + 1):
            for frag in [b_mass, y_mass]:
                yield _mass2mz(frag, cur_charge)


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
