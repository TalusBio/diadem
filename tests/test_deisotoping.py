import numpy as np

from diadem.deisotoping import NEUTRON, deisotope


def test_deisotoping() -> None:
    """Tests that isotopes are collapsed correctly."""
    mz = [
        800.9,
        803.4080,  # Envelope
        804.4108,
        805.4106,
        806.4116,
        810.0,
        812.0,  # Envelope
        812.0 + NEUTRON / 2.0,
    ]
    inten = [1.0, 4.0, 3.0, 2.0, 1.0, 1.0, 9.0, 4.5]
    out_mz, out_inten = deisotope(mz, inten, 2, 5.0, "ppm")
    assert np.allclose(out_inten, np.array([1.0, 10.0, 1.0, 13.5]))
    assert np.allclose(out_mz, np.array([800.9, 803.408, 810.0, 812.0]))
