"""Amino acid masses and other useful mass spectrometry calculations"""
import re


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

    masses = {
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

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    oh = hydrogen + oxygen
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688
    c13_diff = 1.003355

    def __init__(self, masses=None):
        """Initialize the PeptideMasses object"""
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
        seq = re.findall(self._seq_regex, seq)
        calc_mass = sum([self.residue(aa) for aa in seq]) + self.h2o
        for isotope in range(n_isotopes):
            if isotope:
                calc_mass += self.c13_diff

            if charge is not None:
                yield (calc_mass / charge) + self.proton
            else:
                yield calc_mass

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
        max_charge = min(charge, 2)
        seq = re.findall(self._seq_regex, seq)
        for idx in range(1, len(seq) - 1):
            b_mass = sum([self.residue(aa) for aa in seq[:idx]])
            y_mass = sum([self.residue(aa) for aa in seq[idx:]]) + self.h2o
            for cur_charge in range(1, max_charge + 1):
                for frag in [b_mass, y_mass]:
                    yield (frag / cur_charge) + self.proton

    def residue(self, aa):
        """Get the mass of the amino acid or return a modification

        Parameters
        ----------
        aa : str
            The one-letter amino acid or modification string.

        Returns
        -------
        float
            The mass of the amino acid or modification.
        """
        return float(self.masses.get(aa.upper(), aa))
