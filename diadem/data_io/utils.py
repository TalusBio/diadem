from collections.abc import Iterable

import numpy as np
from ms2ml.utils.mz_utils import annotate_peaks
from numpy.typing import NDArray

from diadem.config import MassError


def slice_from_center(center: int, window: int, length: int) -> tuple[slice, int]:
    """Generates a slice provided a center and window size.

    Creates a slice that accounts for the endings of an iterable
    in such way that the window size is maintained.

    Examples
    --------
    >>> my_list = [0,1,2,3,4,5,6]
    >>> slc, center_index = slice_from_center(
    ...     center=4, window=3, length=len(my_list))
    >>> slc
    slice(3, 6, None)
    >>> my_list[slc]
    [3, 4, 5]
    >>> my_list[slc][center_index] == my_list[4]
    True

    >>> slc = slice_from_center(1, 3, len(my_list))
    >>> slc
    (slice(0, 3, None), 1)
    >>> my_list[slc[0]]
    [0, 1, 2]

    >>> slc = slice_from_center(6, 3, len(my_list))
    >>> slc
    (slice(4, 7, None), 2)
    >>> my_list[slc[0]]
    [4, 5, 6]
    >>> my_list[slc[0]][slc[1]] == my_list[6]
    True

    """
    start = center - (window // 2)
    end = center + (window // 2) + 1
    center_index = window // 2

    if start < 0:
        start = 0
        end = window
        center_index = center

    if end >= length:
        end = length
        start = end - window
        center_index = window - (length - center)

    slice_q = slice(start, end)
    return slice_q, center_index


try:
    zip([], [], strict=True)

    def strictzip(*args: Iterable) -> Iterable:
        """Like zip but checks that the length of all elements is the same."""
        return zip(*args, strict=True)

except TypeError:

    def strictzip(*args: Iterable) -> Iterable:
        """Like zip but checks that the length of all elements is the same."""
        # TODO optimize this, try to get the length and fallback to making it a list
        args = [list(arg) for arg in args]
        lengs = {len(x) for x in args}
        if len(lengs) > 1:
            raise ValueError("All arguments need to have the same legnths")
        return zip(*args)


# @profile
def xic(
    query_mz: NDArray[np.float32],
    query_int: NDArray[np.float32],
    mzs: NDArray[np.float32],
    tolerance_unit: MassError = "da",
    tolerance: float = 0.02,
) -> tuple[NDArray[np.float32], list[list[int]]]:
    """Gets the extracted ion chromatogram form arrays.

    Gets the extracted ion chromatogram from the passed mzs and intensities
    The output should be the same length as the passed mzs.

    Returns
    -------
    NDArray[np.float32]
        An array of length `len(mzs)` that integrates such masses in
        the query_int (matching with the query_mz array ...)

    list[list[int]]
        A nested list of length `len(mzs)` where each sub-list contains
        the indices of the `query_int` array that were integrated.

    """
    theo_mz_indices, obs_mz_indices = annotate_peaks(
        theo_mz=mzs,
        mz=query_mz,
        tolerance=tolerance,
        unit=tolerance_unit,
    )

    outs = np.zeros_like(mzs, dtype="float")
    inds = []
    for i in range(len(outs)):
        query_indices = obs_mz_indices[theo_mz_indices == i]
        ints_subset = query_int[query_indices]
        if len(ints_subset) == 0:
            inds.append([])
        else:
            outs[i] = np.sum(ints_subset)
            inds.append(query_indices)

    return outs, inds
