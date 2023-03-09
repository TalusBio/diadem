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
# @profile
def deisotope(
    mz: NDArray[np.float32] | list[float],
    inten: NDArray[np.float32] | list[float],
    max_charge: int,
    diff: float,
    unit: str,
    track_indices: bool = False,
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
    track_indices, bool
        Whether to return the indices of the combined indices as well.

    Examples
    --------
    >>> my_mzs = [800.9, 803.408, 804.4108, 805.4106]
    >>> my_intens = [1-(0.1*i) for i,_ in enumerate(my_mzs)]
    >>> deisotope(my_mzs, my_intens, max_charge=2, diff=5.0, unit="ppm")
    (array([800.9  , 803.408]), array([1. , 2.4]))
    >>> deisotope(my_mzs, my_intens, max_charge=2, diff=5.0, unit="ppm", track_indices=True)
    (array([800.9  , 803.408]), array([1. , 2.4]), ([0], [1, 2, 3]))
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
    for i in range(len(peaks)):
        peaks[i]["indices"] = [i]

    for i in range(len(mz) - 1, -1, -1):
        j = i - 1
        while j >= 0 and mz[i] - mz[j] <= NEUTRON + mass_unit_fun(mz[i], diff):
            delta = mz[i] - mz[j]
            tol = mass_unit_fun(mz[i], diff)
            for charge in range(1, max_charge + 1):
                iso = NEUTRON / charge
                # Note, this assumes that the isotope envelope always decreases in
                # intensity, which is not accurate for high mol weight fragments.
                if abs(delta - iso) <= tol and inten[i] < inten[j]:
                    peaks[j]["intensity"] += peaks[i]["intensity"]
                    if track_indices:
                        peaks[j]["indices"].extend(peaks[i]["indices"])
                    if peaks[i]["charge"] and peaks[i]["charge"] != charge:
                        continue
                    peaks[j]["charge"] = charge
                    peaks[i]["charge"] = charge
                    peaks[i]["envelope"] = j
            j -= 1

    peaks = _filter_peaks(peaks)
    if not track_indices:
        peaks = peaks[0:2]
    return peaks


def _filter_peaks(peaks: dict) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Filters peaks to remove isotope envelopes.

    When passed a list of dictionaries that look like this:
    > {"mz": mz, "intensity": intensity, "envelope": None, "charge": None}
    It filters the ones that are not assigned to be in an envelope,
    thus keeping only monoisotopic peaks.
    """
    peaktuples = [
        (x["mz"], x["intensity"], x["indices"]) for x in peaks if x["envelope"] is None
    ]
    if len(peaktuples) == 0:
        mzs, ints, inds = [], [], []
    else:
        mzs, ints, inds = zip(*peaktuples)
    return np.array(mzs), np.array(ints), inds
