import numpy as np
from numpy.typing import NDArray

NEUTRON = 1.00335


def ppm_to_delta_mass(obs: float, ppm: float) -> float:
    """Converts a ppm error range to a delta mass in th (da?).

    Parameters
    ----------
    obs, float
        observed mass
    ppm, float
        mass range in ppm

    Example
    -------
    ppm_to_delta_mass(1234.567, 50)
    """
    return ppm * obs / 1_000_000.0


# Ported implementation from sage
def deisotope(
    mz: NDArray[np.float32] | list[float],
    inten: NDArray[np.float32] | list[float],
    max_charge: int,
    diff: float,
    unit: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Deisotopes the passed spectra.

    TE MZS NEED TO BE SORTED!!

    Parameters
    ----------
    mz, list[float]
        A list of the mz values to be deisotoped.
    inten, list[float]
        A list of the intensities that correspond in order to the elements in mz.
    max_charge, int
        Maximum charge to look for in the isotopes.
    diff, float
        Tolerance to use when searching (typically 20 for ppm or 0.02 for da)
    unit, str
        Unit for the diff. ppm or da
    """
    if unit.lower() == "da":

        def mass_unit_fun(x: float, y: float) -> float:
            return y

    elif unit.lower() == "ppm":
        mass_unit_fun = ppm_to_delta_mass
    else:
        raise NotImplementedError("Masses need to be either 'da' or 'ppm'")

    peaks = [(mz, intensity) for mz, intensity in zip(mz, inten)]
    peaks = [
        {"mz": mz, "intensity": intensity, "envelope": None, "charge": None}
        for mz, intensity in peaks
    ]

    for i in range(len(mz) - 1, -1, -1):
        j = i - 1
        while j >= 0 and mz[i] - mz[j] <= NEUTRON + mass_unit_fun(mz[i], diff):
            delta = mz[i] - mz[j]
            tol = mass_unit_fun(mz[i], diff)
            for charge in range(1, max_charge + 1):
                iso = NEUTRON / charge
                if abs(delta - iso) <= tol and inten[i] < inten[j]:
                    peaks[j]["intensity"] += peaks[i]["intensity"]
                    if peaks[i]["charge"] and peaks[i]["charge"] != charge:
                        continue
                    peaks[j]["charge"] = charge
                    peaks[i]["charge"] = charge
                    peaks[i]["envelope"] = j
            j -= 1

    peaks = _filter_peaks(peaks)
    return peaks


def _filter_peaks(peaks: dict) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Filters peaks to remove isotope envelopes.

    When passed a list of dictionaries that look like this:
    > {"mz": mz, "intensity": intensity, "envelope": None, "charge": None}
    It filters the ones that are not assigned to be in an envelope,
    thus keeping only monoisotopic peaks.
    """
    peaktuples = [(x["mz"], x["intensity"]) for x in peaks if x["envelope"] is None]
    if len(peaktuples) == 0:
        mzs, ints = [], []
    else:
        mzs, ints = zip(*peaktuples)
    return np.array(mzs), np.array(ints)
