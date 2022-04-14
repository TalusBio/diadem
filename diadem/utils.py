"""Utility Functions"""


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


def mz2int(moverz, precision=4):
    """Convert an m/z to an int

    Parameters
    ----------
    moverz : float
        The m/z value to convert.
    precision : int
        How many decimal places to retain.

    Returns
    -------
    int
        The intergerized m/z value.
    """
    return int(moverz * 10**precision)


def int2mz(mzint, precision=4):
    """Convert an integer to the m/z.

    Parameters
    ----------
    mzint : int
        The integerized m/z value.
    precision : int
        How many decimal places were retained.

    Returns
    -------
    float
        The m/z value.
    """
    return mzint / 10**precision
