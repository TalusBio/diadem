from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from diadem.utilities.neighborhood import _multidim_neighbor_search

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


# @profile
def _deisotope_with_ims_arrays(
    mzs: NDArray[np.float32],
    intensities: NDArray[np.float32],
    imss: NDArray[np.float32] | None = None,
    max_charge: int = 5,
    ims_tolerance: float = 0.01,
    mz_tolerance: float = 0.01,
    track_indices: bool = False,
) -> dict[str, NDArray[np.float32]]:
    """Deisotope a spectrum with IMS data.

    The current implementation allows only absolute values for the ims and mz
    (no ppm).

    This function assumes that the IMS data has already been centroided (collapsed?)
    and that the 3 arrays share ordering.

    If imss is None, will assume there is no IMS dimension in the data.
    """
    if imss is None:
        peak_iter = zip(mzs, intensities)
        peaks = [
            {
                "mz": mz,
                "intensity": intensity,
                "orig_intensity": intensity,
                "envelope": None,
                "charge": None,
            }
            for mz, intensity in peak_iter
        ]
        dim_order = ["mz"]
        extract_values = ["mz", "intensity"]
        spec = {"mz": mzs, "intensity": intensities}
    else:
        peak_iter = zip(mzs, intensities, imss)
        peaks = [
            {
                "mz": mz,
                "ims": ims,
                "intensity": intensity,
                "orig_intensity": intensity,
                "envelope": None,
                "charge": None,
            }
            for mz, intensity, ims in peak_iter
        ]
        dim_order = ["mz", "ims"]
        extract_values = ["mz", "intensity", "ims"]
        spec = {"mz": mzs, "intensity": intensities, "ims": imss}

    dist_funs = {k: lambda x, y: y - x for k in dim_order}
    # sort all elements by their first dimension
    spec_order = np.argsort(spec["mz"])
    peaks = [peaks[i] for i in spec_order]
    spec = {k: v[spec_order] for k, v in spec.items()}

    spec_indices = np.arange(len(spec["mz"]))
    if track_indices:
        extract_values.append("indices")
        for i, peak in enumerate(peaks):
            peak["indices"] = [i]

    # It might be faster to just generate an expanded
    # array with all the charge variants and then do a
    # single search.
    for charge in range(max_charge + 1, 0, -1):
        dist_ranges = {
            "mz": (
                (NEUTRON / charge) - mz_tolerance,
                (NEUTRON / charge) + mz_tolerance,
            ),
            "ims": (-ims_tolerance, ims_tolerance),
        }

        out = _multidim_neighbor_search(
            elems_1=spec,
            elems_2=spec,
            elems_1_indices=spec_indices,
            elems_2_indices=spec_indices,
            dist_funs=dist_funs,
            dist_ranges=dist_ranges,
            dimension_order=dim_order,
        )

        isotope_graph = out.left_neighbors
        for i in np.argsort(-spec["mz"]):
            if i not in isotope_graph:
                continue

            for j in isotope_graph[i]:
                # 0.5 is a magical number ... it is meant to account
                # for the fact that intensity should in theory always increase
                # (except for envelopes of high mass) but should also have some
                # wiggle room for noise.
                intensity_inrange = (
                    0.5 * (peaks[j]["orig_intensity"]) < peaks[i]["orig_intensity"]
                )
                if intensity_inrange and peaks[j]["envelope"] is None:
                    peaks[i]["intensity"] += peaks[j]["intensity"]
                    peaks[j]["charge"] = charge
                    peaks[i]["charge"] = charge
                    peaks[j]["envelope"] = i
                    if track_indices:
                        peaks[i]["indices"].extend(peaks[j]["indices"])
                    # At the end of this, all peaks that belong to an envelope
                    # have a value for "envelope".
                    # Therefore, peaks that need to be filtered out (since their
                    # intensity now bleongs to another peak).

    f_peaks = _filter_peaks(peaks, extract=extract_values)
    f_peaks = dict(zip(extract_values, f_peaks))
    return f_peaks


# Ported implementation from sage
# @profile
def deisotope(
    mz: NDArray[np.float32],
    inten: NDArray[np.float32],
    max_charge: int,
    diff: float,
    unit: str,
    track_indices: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Deisotopes the passed spectra.

    THE MZS NEED TO BE SORTED!!

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
    >>> my_mzs = np.array([800.9, 803.408, 804.4108, 805.4106])
    >>> my_intens = np.array([1-(0.1*i) for i,_ in enumerate(my_mzs)])
    >>> deisotope(my_mzs, my_intens, max_charge=2, diff=5.0, unit="ppm")
    (array([800.9  , 803.408]), array([1. , 2.4]))
    >>> deisotope(my_mzs, my_intens, max_charge=2, diff=5.0, unit="ppm", track_indices=True)
    (array([800.9  , 803.408]), array([1. , 2.4]), ([0], [1, 2, 3]))
    """  # noqa:
    if unit.lower() == "da":
        mz_tolerance = diff

    elif unit.lower() == "ppm":
        # Might give wider tolerances than wanted to the lower end of the values
        # but should be good enough for most cases.
        mz_tolerance = ppm_to_delta_mass(mz.max(), diff)
    else:
        raise NotImplementedError("Masses need to be either 'da' or 'ppm'")

    peaks = _deisotope_with_ims_arrays(
        imss=None,
        mzs=mz,
        intensities=inten,
        max_charge=max_charge,
        mz_tolerance=mz_tolerance,
        track_indices=track_indices,
    )

    return tuple(peaks.values())


# TODO this is a function prototype, if it works abstract it and
# combine with the parent function.


# @profile
def deisotope_with_ims(
    mz: NDArray[np.float32],
    inten: NDArray[np.float32],
    imss: NDArray[np.float32],
    max_charge: int,
    mz_diff: float,
    mz_unit: str,
    ims_diff: float,
    ims_unit: float,
    track_indices: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Deisotopes the passed spectra.

    THE MZS NEED TO BE SORTED!!

    Parameters
    ----------
    mz, list[float]
        A list of the mz values to be deisotoped. (NEEDS to be sorted)
    inten, list[float]
        A list of the intensities that correspond in order to the elements in mz.
    imss, list[float]
        A list of the ims values to use for the search.
    max_charge, int
        Maximum charge to look for in the isotopes.
    mz_diff, float
        Tolerance to use when searching (typically 20 for ppm or 0.02 for da)
    mz_unit, str
        Unit for the diff. ppm or da
    ims_diff, float
        Tolerance to use when searching in the IMS dimension.
    ims_unit, float
        Tolerance unit to use when searching in the ims dimension.
    track_indices, bool
        Whether to return the indices of the combined indices as well.

    Examples
    --------
    >>> my_mzs = np.array([800.9, 803.408, 803.409, 804.4108, 804.4109, 805.4106])
    >>> my_imss = np.array([0.7, 0.7, 0.8, 0.7, 0.8, 0.7])
    >>> my_intens = np.array([1-(0.1*i) for i,_ in enumerate(my_mzs)])
    >>> deisotope_with_ims(my_mzs, my_intens, my_imss, max_charge=2,
    ... mz_diff=5.0, mz_unit="ppm", ims_diff=0.01, ims_unit="abs")
    (array([800.9  , 803.408, 803.409]), array([1. , 2.1, 1.4]), array([0.7, 0.7, 0.8]))
    >>> deisotope_with_ims(my_mzs, my_intens, my_imss, max_charge=2,
    ... mz_diff=5.0, mz_unit="ppm", ims_diff=0.01, ims_unit="abs", track_indices=True)
    (array([800.9  , 803.408, 803.409]), array([1. , 2.1, 1.4]), array([0.7, 0.7, 0.8]), ([0], [1, 3, 5], [2, 4]))
    """  # noqa E501
    if mz_unit.lower() == "da":
        mz_tolerance = mz_diff

    elif mz_unit.lower() == "ppm":
        mz_tolerance = ppm_to_delta_mass(mz.max(), mz_diff)
    else:
        raise NotImplementedError("Masses need to be either 'da' or 'ppm'")

    if ims_unit.lower() == "abs":
        ims_tolerance = ims_diff

    else:
        raise NotImplementedError("only abs is supported as an IMS difference")

    peaks = _deisotope_with_ims_arrays(
        imss=imss,
        mzs=mz,
        intensities=inten,
        max_charge=max_charge,
        mz_tolerance=mz_tolerance,
        ims_tolerance=ims_tolerance,
        track_indices=track_indices,
    )

    return tuple(peaks.values())


def _filter_peaks(
    peaks: list[dict],
    extract: tuple[str],
) -> tuple[NDArray[np.float32] | list, ...]:
    """Filters peaks to remove isotope envelopes.

    When passed a list of dictionaries that look like this:
    > {"mz": mz, "intensity": intensity, "envelope": None, "charge": None}
    It filters the ones that are not assigned to be in an envelope,
    thus keeping only monoisotopic peaks.
    """
    peaktuples = [[x[y] for y in extract] for x in peaks if x["envelope"] is None]
    if len(peaktuples) == 0:
        out_tuple = tuple([] for _ in extract)
    else:
        out_tuple = zip(*peaktuples)
        out_tuple = tuple(
            np.array(x) if y in ["mz", "intensity", "ims"] else x
            for x, y in zip(out_tuple, extract)
        )
    return out_tuple
