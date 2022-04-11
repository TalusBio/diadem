"""Test mass calculations"""
import pytest
import numpy as np
from pyteomics import mass
from diadem.masses import PeptideMasses


def test_precursor():
    """Test precursor mass calculations"""
    seq = "LESLIEK"
    monos = [mass.calculate_mass(seq, charge=z) for z in range(4)]
    print(monos)

    pepcalc = PeptideMasses()
    assert next(pepcalc.precursor(seq)) == pytest.approx(monos[0])
    assert next(pepcalc.precursor(seq, 1)) == pytest.approx(monos[1])
    assert next(pepcalc.precursor(seq, 2)) == pytest.approx(monos[2])
    assert next(pepcalc.precursor(seq, 3)) == pytest.approx(monos[3])


def test_fragments():
    """Test fragment mass calculations"""
    seq = "LESLIEK"
    pepcalc = PeptideMasses()

    pyteomics_frags = np.sort(np.array(list(fragments(seq, maxcharge=1))))
    our_frags = np.sort(np.array(list(pepcalc.fragments(seq, charge=1))))
    np.testing.assert_allclose(pyteomics_frags, our_frags)

    pyteomics_frags = np.sort(np.array(list(fragments(seq, maxcharge=2))))
    our_frags = np.sort(np.array(list(pepcalc.fragments(seq, charge=2))))
    np.testing.assert_allclose(pyteomics_frags, our_frags)


def fragments(peptide, types=("b", "y"), maxcharge=1):
    """This is a function from the pyteomics tutorial."""
    for i in range(1, len(peptide) - 1):
        for ion_type in types:
            for charge in range(1, maxcharge + 1):
                if ion_type[0] in "abc":
                    yield mass.fast_mass(
                        peptide[:i], ion_type=ion_type, charge=charge
                    )
                else:
                    yield mass.fast_mass(
                        peptide[i:],
                        ion_type=ion_type,
                        charge=charge,
                    )
