"""Utilities for scoring."""
import numpy as np
import numba as nb

from . import utils, feature, score_functions


@nb.njit
def score_precursor(
    query_mz: int,
    query_rt: float,
    query_peak: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    frag_array: np.ndarray,
    mz_array: np.ndarray,
    int_array: np.ndarray,
    rt_array: np.ndarray,
    mz_tol: float,
    rt_tol: float,
    min_corr: float,
):
    """Score features"""
    mz_tol = int(mz_tol * query_mz / 1e6)
    mz_bounds = (query_mz - mz_tol, query_mz + mz_tol)
    rt_bounds = (query_rt - rt_tol, query_rt + rt_tol)

    # The retention times:
    prec_rt = utils.unique(rt_array)
    in_rt = slice(
        np.searchsorted(prec_rt, rt_bounds[0], "left"),
        np.searchsorted(prec_rt, rt_bounds[1], "right"),
    )
    prec_rt = prec_rt[in_rt]

    ##############################
    # Step 1: Build the features #
    ##############################
    prec_rows = []
    prec_mz = np.full((len(frag_array), len(prec_rt)), np.nan)
    prec_xic = np.zeros((len(frag_array), len(prec_rt)))

    # To track when multiple m/z might be right.
    used_rt = np.array([False] * len(frag_array))
    curr_rt_idx = 0
    curr_rt = prec_rt[0]

    arrays = zip(mz_array, rt_array, int_array)
    for idx, (mz, rt, int_) in enumerate(arrays):
        used_idx = False
        if rt > rt_bounds[1]:
            break

        if rt < rt_bounds[0]:
            continue

        if rt > curr_rt:
            curr_rt_idx += 1
            curr_rt = rt
            used_rt = np.array([False] * len(frag_array))

        for f_idx, frag in enumerate(frag_array):
            if used_rt[f_idx]:
                continue

            if mz_bounds[0] <= frag <= mz_bounds[1]:
                prec_xic[f_idx, curr_rt_idx] = int_
                prec_mz[f_idx, curr_rt_idx] = mz
                used_rt[f_idx] = True
                if not used_idx:
                    prec_rows.append(idx)
                    used_idx = True

    ################################
    # Step 2: Get correlated peaks #
    ################################
    query_diffs = query_peak - np.mean(query_peak)
    query_rss = np.sum(query_diffs**2)
    areas = np.zeros_like(frag_array)
    features = []
    arrays = zip(frag_array, prec_mz, prec_xic)
    for idx, (frag, mz_vals, int_vals) in enumerate(arrays):
        if np.isnan(mz_vals).all():
            continue

        peak = feature.calc_peak(
            int_array=int_vals,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        corr = pearson_corr(
            x_diffs=query_diffs,
            x_rss=query_rss,
            y_array=peak,
        )

        if corr < min_corr:
            continue

        areas[idx] = feature.integrate(
            peak=peak,
            ret_times=prec_rt,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        mean_mz = np.nanmean(mz_vals)
        feat_info = (
            frag,
            query_rt,
            mean_mz,
            prec_rt,
            int_vals,
            lower_bound,
            upper_bound,
        )
        features.append(feat_info)

    #############################
    # Step 3: Calculate a Score #
    #############################
    score = score_functions.mini_hyperscore(areas)

    return score, features, np.array(prec_rows)


@nb.njit
def pearson_corr(x_diffs, x_rss, y_array):
    """Calcualte the Pearson correlation between two arrays.

    Parameters
    ----------
    x_array : numpy.ndarray of shape (n_peaks,)
        The base array to compare against.
    y_arrays : numpy.ndarray of shape (n_arrays, n_peaks)
        The arrays to calculate the correlation between.

    Returns
    -------
    float
        The Pearson correlation coefficient.
    """
    y_diffs = y_array - np.mean(y_array)
    denom = np.sqrt(x_rss * np.sum(y_diffs**2))
    if denom == 0:
        return 0

    num = np.sum(x_diffs * y_diffs)
    return num / denom
