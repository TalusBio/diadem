from __future__ import annotations

import numpy as np
from numpy import array

# These are just some hard-coded distributions
average_distributions = {
    0: array([1.000]),
    200: array([1.000, 0.108, 0.011, 0.001, 0.000, 0.000]),
    400: array([1.000, 0.219, 0.033, 0.004, 0.000, 0.000, 0.000]),
    600: array([1.000, 0.326, 0.068, 0.011, 0.001, 0.000, 0.000]),
    800: array([1.000, 0.436, 0.116, 0.023, 0.004, 0.000, 0.000, 0.000]),
    1000: array([1.000, 0.536, 0.167, 0.038, 0.007, 0.001, 0.000, 0.000, 0.000]),
    1200: array([1.000, 0.645, 0.238, 0.064, 0.014, 0.003, 0.000, 0.000, 0.000]),
    1400: array([1.000, 0.757, 0.367, 0.133, 0.039, 0.009, 0.002]),
    1600: array([1.000, 0.868, 0.461, 0.181, 0.057, 0.015, 0.003, 0.001]),
    1800: array([1.000, 0.976, 0.565, 0.242, 0.083, 0.024, 0.006, 0.001]),
    2000: array([0.923, 1.000, 0.629, 0.290, 0.107, 0.033, 0.009, 0.002]),
    2200: array([0.837, 1.000, 0.680, 0.336, 0.133, 0.044, 0.013, 0.003, 0.001]),
    2400: array([0.768, 1.000, 0.731, 0.386, 0.163, 0.058, 0.018, 0.005, 0.001]),
    2600: array([0.708, 1.000, 0.784, 0.442, 0.198, 0.074, 0.024, 0.007, 0.002]),
    2800: array([0.662, 1.000, 0.831, 0.494, 0.233, 0.092, 0.031, 0.009, 0.003, 0.001]),
    3000: array([0.617, 1.000, 0.884, 0.557, 0.277, 0.115, 0.041, 0.013, 0.004, 0.001]),
    3200: array([0.578, 1.000, 0.936, 0.622, 0.327, 0.143, 0.054, 0.018, 0.005, 0.001]),
    3400: array(
        [0.543, 1.000, 0.990, 0.692, 0.381, 0.175, 0.069, 0.024, 0.008, 0.002, 0.001]
    ),
    3600: array(
        [0.493, 0.959, 1.000, 0.735, 0.424, 0.203, 0.084, 0.031, 0.010, 0.003, 0.001]
    ),
    3800: array(
        [0.444, 0.913, 1.000, 0.770, 0.464, 0.233, 0.101, 0.038, 0.013, 0.004, 0.001]
    ),
}

max_len = max([len(x) for x in average_distributions.values()])
average_distributions_array = np.array(
    [np.pad(x, (0, max_len - len(x))) for x in average_distributions.values()],
)


def mass_to_dist(mass):
    """Convert a mass to a distribution.

    It is a hard coded-ugly way of doing it that approximates
    to the closest 200 Da.
    Will only work for masses lower than 4000.
    """
    return average_distributions_array[(mass // 200)]


def make_isotope_envelope_vectorized(
    mz,
    charge,
    intensity,
    min_intensity,
    ims: np.ndarray | None = None,
):
    """Make an isotopic envelope for a given mz, charge, and intensity.

    Returns a tuple of (mz, intensity) arrays.

    Examples
    --------
    >>> mzs = np.array([300., 300.])
    >>> ints = np.array([1000., 1000.])
    >>> charges = np.array([1, 2])
    >>> out = make_isotope_envelope_vectorized(mzs, charges, ints, 100)
    >>> out
    (array([300.    , 301.003 , 300.    , 300.5015]), array([1000.,  108., 1000.,  326.]))
    >>> out = make_isotope_envelope_vectorized(mzs, charges, ints, 100, ims=np.array([1.0, 2.0]))
    >>> out
    (array([300.    , 301.003 , 300.    , 300.5015]), array([1000.,  108., 1000.,  326.]), array([1., 1., 2., 2.]))
    """  # noqa: E501
    dist = average_distributions_array[((mz * charge) // 200).astype(int)]
    dist = np.einsum("ij, i -> ij", dist, intensity)

    mz_offset = np.expand_dims(np.arange(dist.shape[1]), axis=-1) * 1.003 / charge
    # shape (num_isotopes, 1)
    mz_dist = (mz_offset + np.expand_dims(mz, axis=0)).T

    intensities = dist.flatten()
    mz_dists = mz_dist.flatten()

    mask = intensities > min_intensity

    mz_dist = mz_dists[mask]
    intensities = intensities[mask]
    if ims is not None:
        ims = np.repeat(ims, dist.shape[1])
        ims = ims[mask]
        return mz_dist, intensities, ims

    return mz_dist, intensities


def simulate_isotopes(
    min_mz,
    max_mz,
    num_peaks,
    min_charge,
    max_charge,
    min_intensity,
    max_intensity,
):
    """Simulate a set of isotopic peaks."""
    mzs = np.random.uniform(min_mz, max_mz, num_peaks)
    ints = np.random.uniform(min_intensity, max_intensity, size=num_peaks)
    charges = np.random.randint(min_charge, max_charge, size=num_peaks)

    mzs, intensities = make_isotope_envelope_vectorized(
        mzs,
        charges,
        ints,
        min_intensity,
    )

    return mzs, intensities


def simulate_isotopes_ims(
    min_mz,
    max_mz,
    num_peaks,
    min_charge,
    max_charge,
    min_intensity,
    max_intensity,
    min_ims,
    max_ims,
):
    """Simulate a set of isotopic peaks."""
    mzs = np.random.uniform(min_mz, max_mz, num_peaks)
    ints = np.random.uniform(min_intensity, max_intensity, size=num_peaks)
    charges = np.random.randint(min_charge, max_charge, size=num_peaks)

    mzs, intensities = make_isotope_envelope_vectorized(
        mzs,
        charges,
        ints,
        min_intensity,
    )

    return mzs, intensities


def _split_ims(ims: float, intensity: float, ims_std: float, ims_binwidth=0.002):
    """Split an ims peak into multiple peaks.

    The number of peaks is the integer square root of the intensity.

    The new ims is drawn from a normal distribution with a mean of the
    provided ims, with the provided standard deviation.

    The intensity of each peak is drawn from a uniform distribution between 0
    and 2*(intensity)/len(num of peaks).
    """
    ims_out = np.random.normal(ims, ims_std, size=int(np.sqrt(intensity)) + 1)
    ims_intensity = np.random.uniform(
        0,
        2 * (intensity) / len(ims_out),
        size=len(ims_out),
    )
    ims_intensity_out = np.histogram(
        ims_out,
        bins=np.arange(ims.min(), ims.max(), ims_binwidth),
        weights=ims_intensity,
    )
    return ims_out, ims_intensity_out


def extend_ims(seed_mzs, intensities, ims_start=0.7, ims_end=1.2, std_ims=0.01):
    """Extend a set of peaks to include IMS coordinates.

    A random ims is assigned to every peak and a distribution of peaks is generated
    splitting the intensity of the peak between the new peaks. (the new distribution
    is random so the total might not be the same).

    The IMS coordinates are drawn from a normal distribution with a mean of the
    average IMS value and a standard deviation of the standard deviation of the
    IMS values.

    Returns
    -------
    mzs : np.ndarray
        The m/z values of the peaks.
    intensities : np.ndarray
        The intensities of the peaks.
    ims : np.ndarray
        The IMS values of the peaks.
    indices : np.ndarray
        This indices correspond to the originally passed seed mzs and intensities.
    """
    num_ims = len(seed_mzs)
    ims = np.random.uniform(ims_start, ims_end, size=num_ims)
    ims_inten_pairs = [
        _split_ims(im, inten, ims_std=std_ims, ims_binwidth=0.002)
        for im, inten in zip(ims, intensities)
    ]

    out_imss = []
    out_intens = []
    out_mzs = []
    out_indices = []

    for i, (imss, intens, seed_mz) in enumerate(zip(*ims_inten_pairs, seed_mzs)):
        out_imss.extend(imss)
        out_intens.extend(intens)
        out_mzs.extend([seed_mz] * len(imss))
        out_indices.extend([i] * len(imss))

    return (
        np.array(out_mzs),
        np.array(out_intens),
        np.array(out_imss),
        np.array(out_indices),
    )


def simulate_ims_isotopes(  # noqa: D103
    min_mz,
    max_mz,
    num_peaks,
    min_charge,
    max_charge,
    min_intensity,
    max_intensity,
    min_ims,
    max_ims,
):
    raise NotImplementedError


def add_noise(values, snr):
    """Add noise to a set of peaks.

    The noise is added by adding a random number to each peak. The random number is
    drawn from a normal distribution with a standard deviation equal to the intensity
    of the peak divided by the SNR.
    """
    noise = np.random.normal(0, values / snr)
    return values + noise


def jitter_values(values, std):
    """Jitter a set of values.

    The values are jittered by adding a random number to each value. The random number
    is drawn from a normal distribution with a standard deviation equal to the standard
    deviation of the values.
    """
    noise = np.random.normal(0, std, size=len(values))
    return values + noise


def get_noise_peaks(mz_min, mz_max, ints, quantile, pct):
    """Generates noise peaks from the distribution of intensities.

    Adds uniformly distributed peaks in the given mz range that do not belong to
    an isotope envelope. Using the lowest quantile of the intensity distribution,
    a threshold is determined below which no peaks are added.
    """
    noise_intensity = np.quantile(ints, quantile)
    noise_mzs = np.random.uniform(mz_min, mz_max, size=int(len(ints) * pct))
    noise_ints = np.random.uniform(0, noise_intensity, size=len(noise_mzs))

    return noise_mzs, noise_ints


# clean simple spectrum
def clean_simple_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean simple spectrum.

    Returns
    -------
    mzs : np.ndarray
        The m/z values of the peaks.
    intensities : np.ndarray
        The intensities of the peaks.
    """
    mzs, ints = simulate_isotopes(1000, 2000, 100, 1, 5, 1000, 10000)
    ints = add_noise(ints, 100)
    return mzs, ints


# clean complicated spectrum
def clean_complicated_spectrum():
    """Generate a clean simple spectrum.

    Returns
    -------
    mzs : np.ndarray
        The m/z values of the peaks.
    intensities : np.ndarray
        The intensities of the peaks.
    """
    mzs, ints = simulate_isotopes(1000, 2000, 100, 1, 5, 1000, 10000)
    ints = add_noise(ints, 100)
    mzs = jitter_values(mzs, 0.01)
    mzs2, ints2 = get_noise_peaks(1000, 2000, ints, 0.1, 0.1)
    mzs = np.concatenate([mzs, mzs2])
    ints = np.concatenate([ints, ints2])
    return mzs, ints


# noisy simple spectrum
def noisy_simple_spectrum():
    """Generate a noisy simple spectrum.

    Returns
    -------
    mzs : np.ndarray
        The m/z values of the peaks.
    intensities : np.ndarray
        The intensities of the peaks.
    """
    raise NotImplementedError


# noisy complicated spectrum
def noisy_complicated_spectrum():
    """Generate a noisy complicated spectrum.

    Returns
    -------
    mzs : np.ndarray
        The m/z values of the peaks.
    intensities : np.ndarray
        The intensities of the peaks.
    """
    npeaks = 5_000

    mzs, ints = simulate_isotopes(
        1000,
        2000,
        npeaks,
        1,
        5,
        min_intensity=1_000,
        max_intensity=100_000,
    )
    raise NotImplementedError


if __name__ == "__main__":
    out = simulate_isotopes(1000, 2000, 3, 1, 2, 1000, 10000)
