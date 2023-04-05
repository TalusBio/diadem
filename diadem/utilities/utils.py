import gc
from contextlib import contextmanager

import numpy as np
import uniplot
from loguru import logger
from ms2ml import Peptide
from numpy.typing import NDArray


def plot_to_log(*args, **kwargs) -> None:  # noqa
    """Plot to log.

    Generates a plot of the passed data to the function.
    All arguments are passed internally to uniplot.plot_to_string.
    """
    for line in uniplot.plot_to_string(*args, **kwargs):
        logger.debug(line)


@contextmanager
# @profile
def disabled_gc() -> None:
    """Context manager that disables the garbage collector.

    Once the context manager finishes, it makes a full collection and
    enables again the GC

    Usage:
    >   with disabled_gc():
    >       # Do something where you dont want the gc enabled

    """
    try:
        logger.debug("Disabling GC")
        gc.disable()
        yield
    finally:
        logger.debug("Collecting GC")
        gc.collect(0)
        logger.debug("Enabling GC")
        gc.enable()


# From https://datagy.io/python-split-list/
def chunk_array(arr: NDArray, chunksize: int) -> list[NDArray]:
    """Splits an array into chunks of a given size.

    Parameters
    ----------
    arr : NDArray
        The array to split
    chunksize : int
        The size of the chunks


    Examples
    --------
    >>> tmp = np.arange(5)
    >>> chunk_array(tmp, 2)
    [array([0, 1]), array([2, 3]), array([4])]
    """
    chunked_list = []
    for i in range(0, len(arr), chunksize):
        chunked_list.append(arr[i : i + chunksize])
    return chunked_list


def make_decoy(pep: Peptide) -> Peptide:
    """Makes a decoy peptide.

    Parameters
    ----------
    pep : Peptide
        The peptide to make a decoy of

    Examples
    --------
    >>> pep = Peptide.from_sequence("LESLIEK/2")
    >>> pep.to_proforma()
    'LESLIEK/2'
    >>> make_decoy(pep).to_proforma()
    'LEILSEK/2'
    """
    seq = pep.sequence
    charge = pep.charge
    seq = seq[:1] + list(reversed(seq[1:-1])) + seq[-1:]
    properties = pep.properties
    pep = Peptide(seq, properties=properties, config=pep.config, extras=pep.extras)
    assert pep.charge == charge
    return pep


def get_slice_inds(arr: NDArray, minval: float, maxval: float) -> slice:
    """Gets the slide indices that include a range.

    Parameters
    ----------
    arr : NDArray
        A sorted 1d array
    minval : float
        The minimum value to include
    maxval : float
        The maximum value to include
    is_ends : bool
        Whether the slice should include the first index

    Examples
    --------
    >>> arr = np.array(range(500)) / 100
    >>> minval = 2.3
    >>> maxval = 3.8
    >>> get_slice_inds(arr, minval, maxval)
    slice(229, 381, None)
    >>> arr = np.array(range(500))
    >>> slc = get_slice_inds(arr, minval, maxval)
    >>> slc
    slice(2, 4, None)
    >>> arr[slc]
    array([2, 3])
    """
    slice_min = np.searchsorted(arr, minval, side="left") - 1
    slice_min = max(0, slice_min)

    # slice_max = np.searchsorted(arr[slice_min:], maxval, side="right")
    # slice_max = slice_min + slice_max
    i = 0
    for i, val in enumerate(arr[slice_min:]):
        if val > maxval:
            break

    slice_max = slice_min + i
    return slice(slice_min, slice_max)


def is_sorted(a: NDArray) -> bool:
    """Checks if an array is sorted."""
    return np.all(a[:-1] <= a[1:])


def check_sorted(a: NDArray) -> None:
    """Raises an error if the array is not sorted."""
    if not is_sorted(a):
        raise RuntimeError("Array expected to be sorted is not")
