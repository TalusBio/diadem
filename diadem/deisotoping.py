from math import factorial as fact

import numpy as np
from ms2ml.constants import PROTON
from numpy.typing import NDArray

EPS = 1e-8
CUTOFF = 1
NUM_ISOTOPES = 5


def calc_averagine_dist(uncharged_mass: int | float, num_isotopes: int) -> list[float]:
    n_averagines = uncharged_mass / 111.1244
    n_atoms = round(15.5734 * n_averagines)

    # Coefficient derived by the weighted average of +x isotopes
    # using the averagine composition and the isotopic distribution
    # of all corresponding atoms.
    p = 0.00439
    out = [
        (fact(n_atoms) / (fact(k) * fact(n_atoms - k)))
        * ((p**k) * ((1 - p) ** (n_atoms - k)))
        for k in range(num_isotopes)
    ]
    return out


def test_calc_averagine() -> None:
    # LESLIEK; mass = 830.45
    # Expected is 61.7% monoisotopic, 45.9% of that +1, 12.9% +2, 2.699% +3, 0.4% +4
    dist = calc_averagine_dist(830.45, num_isotopes=5)
    expected_dist = 0.617 * np.array([1, 0.459, 0.129, 0.0699, 0.004])

    assert np.allclose(dist, expected_dist, rtol=0.05, atol=0.05)


class IntegerAveragines:
    """Calculates averagine distributions.

    Implements a singleton class that calculates the averagine distributions of mass
    and caches the results.

    For computational efficiency it:
        1. Greedily computes the cache of all masses from 100 to 5100 Da
        1. When asked to calculate a distribution it uses the integer of the passed mass.
        1. It only does the first 5 isotopic peaks (set by the NUM_ISOTOPES constant)
    """

    cache = {
        i + 100: calc_averagine_dist(i + 100, num_isotopes=NUM_ISOTOPES)
        for i in range(5000)
    }

    @classmethod
    def calc_distribution(cls, mass: float | int) -> list[float]:
        """Returns the isotope distribution for a mass.

        Check the class documentation for details.
        """
        mass = int(mass)
        try:
            out = cls.cache[mass]
        except KeyError:
            cls.cache[mass] = calc_averagine_dist(mass, num_isotopes=NUM_ISOTOPES)
            out = cls.cache[mass]

        return out


def deisotope(
    mzs: NDArray[np.float32],
    intensities: NDArray[np.float32],
    charges: NDArray[np.int32] = np.arange(1, 4),
    max_dm: float = 0.02,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    isotopes = np.arange(-1, NUM_ISOTOPES)
    flat_charges = np.repeat(charges, len(isotopes))
    flat_isotopes = np.repeat(
        isotopes.reshape(1, len(isotopes)), len(charges), axis=0
    ).flatten()

    flat_deltamasses = PROTON * flat_isotopes / flat_charges

    dm_order = np.argsort(flat_deltamasses)
    ordered_deltamasses = flat_deltamasses[dm_order]

    gi = 0
    out = []

    for i, mz in enumerate(mzs):
        matching = [[] for _, _ in enumerate(ordered_deltamasses)]

        for dmi, dm in enumerate(ordered_deltamasses):
            for ii, mz2 in enumerate(mzs[gi:], start=gi):
                calc_dm = mz2 - (
                    mz + dm
                )  # low when mz is 2 big, high when mz2 is 2 big
                if calc_dm < -max_dm:
                    gi = ii
                elif calc_dm > max_dm:
                    break
                else:
                    matching[dmi].append(ii)

        out.append(matching)

    tmp = np.zeros((len(intensities), *flat_deltamasses.shape))

    for i, iv in enumerate(out):
        for j, jv in enumerate(iv):
            nj = dm_order[j]
            if len(jv) == 0:
                tmp[i, nj] = 0
            else:
                tmp[i, nj] = np.sum(intensities[jv])

    reshaped_summed_ints = tmp.reshape((len(intensities), len(charges), len(isotopes)))
    # now shape is [num_peaks, num_charges, num_isotopes]

    norm_factor = np.einsum("mci -> mc", reshaped_summed_ints + EPS)
    normed_summed_ints = (reshaped_summed_ints + EPS) / np.expand_dims(norm_factor, -1)

    expected_isotopes = _calculate_isotopes_array(mzs=mzs, charges=charges)
    # expected isotopes is a tensor of shape [len(mzs), len(charges), len(isotopes)]

    # Kullback–Leibler Divergence
    KL_divergence = (
        expected_isotopes
        * np.log((expected_isotopes + EPS) / (normed_summed_ints + EPS))
    ).sum(axis=-1)
    keep = KL_divergence < CUTOFF

    all_used = set()

    new_mzs = []
    new_intensities = []

    for mzi, ci in zip(*np.where(keep)):
        # TODO cache the indices for each charge
        charge = charges[ci]
        extract_indices = np.where(charge == flat_charges[dm_order])[0]
        cluster_indices = []

        for ei in extract_indices:
            spectrum_indices = out[mzi][ei]
            cluster_indices.extend(spectrum_indices)

        all_used.update(set(cluster_indices))
        mono_mass = mzs[mzi]
        cluster_intensity = intensities[list(set(cluster_indices))].sum()

        new_mzs.append(mono_mass)
        new_intensities.append(cluster_intensity)

    all_unused = set(range(len(mzs))) - all_used

    unused_indices = list(all_unused)

    new_mzs = np.concatenate([np.array(new_mzs), mzs[unused_indices]])
    new_order = np.argsort(new_mzs)

    new_mzs = new_mzs[new_order]
    new_intensities = np.concatenate(
        [np.array(new_intensities), intensities[unused_indices]]
    )[new_order]

    return new_mzs, new_intensities


def test_deisotoping() -> None:
    """Tests that deisotoping works.

    The data used is this...

    Name: YLRDVNC[160]PFK/3
    MW: 1313.6660
    PrecursorMZ: 437.8887
    619.3560	392.0	a5/-0.000
    630.3398	418.2	b5-17/0.015 <- Monoisotopic
    631.3586	454.2	b5-17i/0.034 << Isotope
    632.3532	95.6	b5-17i/0.029 << Isotope
    647.3430	1473.1	b5/-0.008 <- Monoisotopic
    648.3577	7303.5	b5i/0.007 << Isotope
    649.3605	2029.0	b5i/0.009 << Isotope
    665.3113	171.6	y5/0.004
    733.3970	138.5	a6/-0.002

    So the final out should be 5 peaks
    """
    tinyspec_mzs = np.array(
        [
            619.3560,
            630.3398,
            631.3586,
            632.3532,
            647.3430,
            648.3577,
            649.3605,
            665.3113,
            733.3970,
        ]
    )

    tinyspec_intensities = np.array(
        [
            392.0,
            418.2,
            454.2,
            95.6,
            1473.18,
            7303.57,
            2029.09,
            171.6,
            138.5,
        ]
    )
    out = deisotope(tinyspec_mzs, tinyspec_intensities)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(len(x) == 5 for x in out)


def kld(
    expected_isotopes: NDArray[np.float32], observed_isotopes: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Calculates a divergence metric for two distributions.

    The calcuated metric is the Kullback–Leibler Divergence.
    Higher values mean a more divergent distribution.

    Details
    -------
    Provided two discrete distributions of length K
    and X number of observations. This functio takes two tensors of
    shape [X, K] to be compared and returns K metrics.

    The range of the metric is [0, Inf]
    Note that the metric is not symetrical!

    Example:
    -------
    >>> x = np.array([[0.1, 0.2, 0.7]])
    >>> y = np.array([[0.33, 0.33, 0.33]])
    >>> y.shape
    (1, 3)
    >>> kld(x, y)
    array([0.30684407])

    >>> x = np.array([[0.1, 0.9], [0.9, 0.1]])
    >>> y = np.array([[0.2, 0.8], [0.2, 0.8]])
    >>> kld(x, y)
    array([0.03669002, 1.14572548])

    """
    # Kullback–Leibler Divergence
    log_ratios = np.log((expected_isotopes + EPS) / (observed_isotopes + EPS))
    out = (expected_isotopes * log_ratios).sum(axis=-1)

    return out


def _calculate_isotopes_array(
    mzs: NDArray[np.float32], charges: NDArray[np.int32]
) -> NDArray[np.float32]:
    # Since each isotope dist is given as an array of shape [num_isotopes], this
    # generates an array fo shape [mzs_length, num_charges, num_isotopes]
    # Adding the [0] is the expected abundance of the -1 isotope
    expected_isotopes = []
    for mz in mzs:
        mz_isotopes = []
        for c in charges:
            mzc_isotopes = [0] + IntegerAveragines.calc_distribution(mz * c)
            mz_isotopes.append(mzc_isotopes)
        expected_isotopes.append(mz_isotopes)

    expected_isotopes = np.array(expected_isotopes)
    return expected_isotopes
