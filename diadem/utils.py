"""Utility Functions"""
import numba as nb
import numpy as np


def groupby_max(df, by_cols, max_col):
    """Quickly get the indices for the maximum value of col"""
    by_cols = listify(by_cols)
    idx = (
        df.sample(frac=1)
        .sort_values(by_cols + [max_col], axis=0)
        .drop_duplicates(by_cols, keep="last")
        .index
    )

    return idx


def listify(obj):
    """Turn an object into a list, but don't split strings"""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)


def bools2bytes(bool_array):
    """Convert an array of booleans to a Bytes object.

    From: https://stackoverflow.com/questions/32675679/
    convert-binary-string-to-bytearray-in-python-3

    Parameters
    ----------
    bool_array : list of bool
        The array of booleans to convert to a Bytes object.

    Returns
    -------
    Bytes
        The encoded bitstring
    """
    bitstring = "".join(["1" if b else "0" for b in bool_array])
    bits = int(bitstring, 2)
    return bits.to_bytes((len(bitstring) + 7) // 8, byteorder="big")


def bytes2bools(byte_obj):
    """Convert a Bytes object back to an array of booleans.

    Parameters
    ----------
    byte_obj : Bytes
        The Bytes object to decode.

    Returns
    -------
    list of bools
        The array of booleans.
    """
    bitstring = bin(int.from_bytes(byte_obj, byteorder="big"))
    return [b == "1" for b in bitstring[2:]]


@nb.njit
def unique(srt_array: np.ndarray) -> np.ndarray:
    """Get the unique values of a sorted numpy array.

    Parameters
    ----------
    srt_array : numpy.ndarray
        A sorted 1D numpy array.

    Returns
    -------
    numpy.ndarray
    """
    vals = []
    for val in srt_array:
        if not vals:
            vals.append(val)
            continue

        if val == vals[-1]:
            continue

        vals.append(val)

    return np.array(vals)
