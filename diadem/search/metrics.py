import numpy as np
from numpy.typing import NDArray


def cosinesim(x: NDArray, y: NDArray) -> NDArray:
    """Computes the cosine similarity between two vectors.

    The function compues the similarity along the last axis.
    Values closer to 1 mean that the vectors are more similar.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([1.1, 2.2, 2.9])
    >>> round(cosinesim(x, y), 6)
    0.998016
    >>> x = np.sin(np.arange(200)/30)
    >>> y = np.cos(np.arange(200)/30)
    >>> round(cosinesim(x, y), 6)
    0.019283
    """
    EPS = 1e-5  # noqa
    x, y = x / (x.max() + EPS), y / (y.max() + EPS)
    out = np.dot(x, y)
    out /= np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + EPS
    return out


def max_rolling(a, window, axis=1):
    """From this answer:
    https://stackoverflow.com/a/52219082.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.max(rolling, axis=axis)


def spectral_angle(x: NDArray, y: NDArray) -> NDArray:
    """Computes the spectral angle between two vectors.

    Values closer to 1 mean that the vectors are more similar.

    It is pretty much the same as a cosine sim but more astringent,
    it is harder to get to values closer to 1.

    The range of the function is (0,1)

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([1.1, 2.2, 2.9])
    >>> round(spectral_angle(x, y), 6)
    0.959892
    >>> x = np.sin(np.arange(200)/30)
    >>> y = np.cos(np.arange(200)/30)
    >>> round(spectral_angle(x, y), 6)
    0.012276
    """
    return 1 - (2 * (np.arccos(cosinesim(x, y)) / np.pi))


def get_ref_trace_corrs(arr: NDArray[np.float32], ref_idx: int) -> NDArray[np.float32]:
    """Gets the correlation of all elements to a reference trace.

    Parameters
    ----------
    arr: NDArray
        An array of shape [..., mz, rt], so the trace is considered
        along the rt axis. (the reference trace would be [..., idx, rt])

    ref_idx: int
        index to use in the array as the reference trace


    Examples
    --------
    >>> x = np.stack([np.cos((0.1*x)+ (np.arange(20)/20)) for x in range(10)], axis = 0)
    >>> x.shape
    (10, 20)

    This would be equivalent to a stack of 10 mzs and 20 spectra.

    >>> out = get_ref_trace_corrs(x, 5)
    >>> [round(x, 4) for x in out]
    [0.8355, 0.8597, 0.8869, 0.9182, 0.9551, 0.9989, 0.9436, 0.8704, 0.7722, 0.6385]
    """
    arr = max_rolling(arr, 3, axis=1)
    norm = np.linalg.norm(arr + 1e-5, axis=-1)
    normalized_arr = arr / np.expand_dims(norm, axis=-1)
    ref_trace = normalized_arr[..., ref_idx, ::1]
    # ref_trace = np.stack([ref_trace, ref_trace[..., ::-1]]).min(axis=0)
    # ref_trace = np.stack([ref_trace, ref_trace[..., ::-1]]).min(axis=0)
    spec_angle_weights = spectral_angle(
        normalized_arr.astype("float"),
        ref_trace.astype("float"),
    )

    return spec_angle_weights
