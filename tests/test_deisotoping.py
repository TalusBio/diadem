import numpy as np
from numpy import array  # noqa: F401

from diadem.deisotoping import NEUTRON, deisotope

single_clean_envelope_text = """
{
    "mz": array(
        [470.2321, 469.5762, 469.2416, 469.9097,
         470.2435, 470.5771, 467.7124, 467.2108, 471.6956, ]
    ),
    "ims": array(
        [0.77803, 0.79777, 0.79688, 0.79611, 0.79404,
         0.79568, 0.80016, 0.79466, 0.81096, ]
    ),
    "intensity": array(
        [92.0, 13729.0, 18491.0, 7138.0, 3081.0,
         1739.0, 751.0, 356.0, 293.0]
    ),
}
"""

single_clean_envelope = eval(single_clean_envelope_text)


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
    mz = np.array(mz)
    inten = np.array([1.0, 4.0, 3.0, 2.0, 1.0, 1.0, 9.0, 4.5])
    out_mz, out_inten = deisotope(mz, inten, 2, 5.0, "ppm")
    assert np.allclose(out_inten, np.array([1.0, 10.0, 1.0, 13.5]))
    assert np.allclose(out_mz, np.array([800.9, 803.408, 810.0, 812.0]))
