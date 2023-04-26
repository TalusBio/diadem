"""Contians and implements ways to look for and represent neighbors.

This module contains implementations related to finding "neighbors"
and representations for those neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from diadem.utilities.utils import is_sorted


@dataclass
class IndexBipartite:
    """Simple bipartite graph structure.

    This dataclass is meant to contain connections between indices
    in other structures.

    Examples
    --------
    >>> bg = IndexBipartite()
    >>> bg.add_connection(1,2)
    >>> bg.add_connection(1,3)
    >>> bg.add_connection(2,2)
    >>> bg
    IndexBipartite(left_neighbors={1: [2, 3], 2: [2]},
      right_neighbors={2: [1, 2], 3: [1]})
    >>> bg.get_neighbor_indices()
    (array([1, 2]), array([2, 3]))
    """

    left_neighbors: dict = field(default_factory=dict)
    right_neighbors: dict = field(default_factory=dict)

    def add_connection(self, left: int, right: int) -> None:
        """Adds a connection to the graph.

        See the class docstring for details.
        """
        self.left_neighbors.setdefault(left, []).append(right)
        self.right_neighbors.setdefault(right, []).append(left)

    def get_neighbor_indices(self) -> tuple[NDArray, NDArray]:
        """Returns the neighborhoods as two arrays.

        This returns two arrays that represent all indices tht have
        a neghbor in their corresponding counterpart.

        Returns
        -------
        x, y: tuple[NDArray, NDAarray]
            an array representing the indices that have a neighbor.
            In other words, all elements with index xi will have a
            neighbor in y.

            Every element in x and every element in y is unique.

        Examples
        --------
        >>> bg = IndexBipartite()
        >>> bg.add_connection(1,1)
        >>> bg.add_connection(1,2)
        >>> bg.add_connection(1,3)
        >>> bg.add_connection(2,2)
        >>> bg.get_neighbor_indices()
        (array([1, 2]), array([1, 2, 3]))
        """
        x = np.sort(np.array(list(self.left_neighbors)))
        y = np.sort(np.array(list(self.right_neighbors)))
        return x, y

    def get_matching_indices(self) -> tuple[NDArray, NDArray]:
        """Returns the matching indices in the neighborhood as two arrays.

        Note: this differs from the `get_neighbor_indices` function
        because it assures that the lengths of both arrays are the same.
        Therefore

        This returns two arrays that represent all indices tht have
        a neghbor in their corresponding counterpart.

        Returns
        -------
        x,y: tuple[NDArray, NDArray]
            Returns two arrays where every element i in array x is
            a neighbor of element i in array y.

            This entails that there will be duplicates if for instance
            an element in x has many neighbors in y.

        Examples
        --------
        >>> bg = IndexBipartite()
        >>> bg.add_connection(1,1)
        >>> bg.add_connection(1,2)
        >>> bg.add_connection(1,3)
        >>> bg.add_connection(2,2)
        >>> bg.get_matching_indices()
        (array([1, 1, 1, 2]), array([1, 2, 3, 2]))
        """
        out_x = []
        out_y = []

        for x in self.left_neighbors.keys():
            for xn in self.left_neighbors[x]:
                out_x.append(x)
                out_y.append(xn)

        return np.array(out_x), np.array(out_y)

    @classmethod
    def from_arrays(cls, x: NDArray, y: NDArray) -> IndexBipartite:
        """Builds a bipartite index from two arrays.

        Arguments:
        ---------
        x, y: NDArray
            Two arrays of the same length that contain the indices
            that match between them. For example element i in the
            array y is a neighbor of element i in array x.

        Examples
        --------
        >>> a = np.array([1,1,1,51])
        >>> b = np.array([2,3,4,1])
        >>> IndexBipartite.from_arrays(a,b)
        IndexBipartite(left_neighbors={1: [2, 3, 4], 51: [1]},
            right_neighbors={2: [1], 3: [1], 4: [1], 1: [51]})
        """
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")

        new = cls()
        for x1, y1 in zip(x, y):
            new.add_connection(x1, y1)

        return new

    def intersect(self, other: IndexBipartite) -> IndexBipartite:
        """Returns the intersection of two bipartite graphs.

        Arguments:
        ---------
        other: IndexBipartite
            The other bipartite graph to intersect with.

        Returns
        -------
        IndexBipartite
            A new bipartite graph that contains the intersection of
            both graphs.

        Examples
        --------
        >>> bg1 = IndexBipartite()
        >>> bg1.add_connection(1,1)
        >>> bg1.add_connection(1,2)
        >>> bg1.add_connection(1,3)
        >>> bg1.add_connection(1,4)
        >>> bg1.add_connection(2,2)
        >>> bg2 = IndexBipartite()
        >>> bg2.add_connection(1,1)
        >>> bg2.add_connection(1,2)
        >>> bg2.add_connection(1,3)
        >>> bg2.add_connection(2,2)
        >>> bg2.add_connection(2,3)
        >>> bg1.intersect(bg2)
        IndexBipartite(left_neighbors={1: [1, 2, 3], 2: [2]},
            right_neighbors={1: [1], 2: [1, 2], 3: [1]})
        """
        new = IndexBipartite()

        for x in self.left_neighbors.keys():
            if x in other.left_neighbors:
                for xn in self.left_neighbors[x]:
                    if xn in other.left_neighbors[x]:
                        new.add_connection(x, xn)

        return new


def _default_dist_fun(x: float, y: float) -> float:
    """Default distance function to use."""
    return y - x


@dataclass
class NeighborFinder:
    """Class to find neighbors in a multidimensional space.

    This class is used to find neighbors in a multidimensional space.

    Parameters
    ----------
    dist_ranges: dict[str, tuple[float, float]]
        A dictionary that contains the ranges of the distances
        for each dimension. The keys of the dictionary are the
        names of the dimensions and the values are tuples that
        contain the minimum and maximum distance for that dimension.
    dist_funs: dict[str, callable]
        A dictionary that contains the distance functions for each
        dimension. The keys of the dictionary are the names of the
        dimensions and the values are the distance functions.
        The distance functions must take two arguments and return
        a float.
    order: tuple[str]
        The order in which the dimensions should be searched.
        If this is None then the order will be the same as the
        order of the keys in the dist_ranges dictionary.
    force_vectorized: bool
        This forces the search to use a vectorized implementation that
        might be faster depending on the use case, in theory does more
        operations but those operations happen in cpu cache. (test it ...)
    """

    dist_ranges: dict[str, tuple[float, float]]
    dist_funs: dict[str, callable]
    order: tuple[str]
    force_vectorized: bool

    def __post_init__(self) -> None:
        """Post init function."""
        if self.order is None:
            self.order = tuple(self.dist_ranges.keys())

        if self.dist_funs is None:
            self.dist_funs = {k: _default_dist_fun for k in self.dist_ranges}

        if not set(self.dist_ranges.keys()) == set(self.dist_funs.keys()):
            raise ValueError("dist_ranges and dist_funs must have the same keys")

    def find_neighbors(
        self,
        elems1: dict[str, NDArray],
        elems2: dict[str, NDArray],
    ) -> IndexBipartite:
        """Finds neighbors in a multidimensional space."""
        out = multidim_neighbor_search(
            elems1=elems1,
            elems2=elems2,
            dist_ranges=self.dist_ranges,
            dist_funs=self.dist_funs,
            order=self.order,
            force_vectorized=self.force_vectorized,
        )
        return out


# @profile
def find_neighbors_sorted(
    x: NDArray,
    y: NDArray,
    dist_fun: callable,
    low_dist: float,
    high_dist: float,
    allowed_neighbors: dict[int, set[int]] | None = None,
) -> IndexBipartite:
    """Finds neighbors between to sorted arrays.

    Parameters
    ----------
    x, NDArray:
        First array to use to find neighbors
    y, NDArray:
        Second array to use to find neighbors
    dist_fun:
        Function to calculate the distance between an element
        in `x` and an element in `y`.
        Note that this asumes directionality and should increase in value.
        In other words ...
        dist_fun(x[0], y[0]) < dist_fun(x[0], y[1]); assuming that y[1] > y[0]
    low_dist:
        Lowest value allowable as a distance for two elements
        to be considered neighbors
    low_dist:
        Highest value allowable as a distance for two elements
        to be considered neighbors
    high_dist:
        Highest value allowable as a distance for two elements.
    allowed_neighbors:
        A dictionary that contains the allowed neighbors for each
        element in `x`. The keys of the dictionary are the indices
        of the elements in `x` and the values are sets that contain
        the indices of the elements in `y` that are allowed to be
        neighbors of the element in `x`. If this is None then all
        elements in `y` are allowed to be neighbors of the elements
        in `x`.


    Examples
    --------
    >>> x = np.array([1.,2.,3.,4.,5.,15.,25.])
    >>> y = np.array([1.1, 2.3, 3.1, 4., 25., 25.1])
    >>> dist_fun = lambda x,y: y - x
    >>> low_dist = -0.11
    >>> high_dist = 0.11
    >>> find_neighbors_sorted(x,y,dist_fun,low_dist, high_dist)
    IndexBipartite(left_neighbors={0: [0], 2: [2], 3: [3], 6: [4, 5]},
      right_neighbors={0: [0], 2: [2], 3: [3], 4: [6], 5: [6]})
    >>> find_neighbors_sorted(x,y,dist_fun,low_dist, high_dist, allowed_neighbors={6: {5, 4}})
    IndexBipartite(left_neighbors={6: [4, 5]}, right_neighbors={4: [6], 5: [6]})
    """  # noqa: E501
    assert is_sorted(x)
    assert is_sorted(y)
    assert low_dist < high_dist
    assert dist_fun(low_dist, high_dist) > 0
    assert dist_fun(high_dist, low_dist) < 0

    neighbors = IndexBipartite()

    ii = 0

    if allowed_neighbors is None:
        iter_x = range(len(x))
    else:
        iter_x = sorted(allowed_neighbors.keys())

    for i in iter_x:
        x_val = x[i]
        # if (abs(x_val - 0.8112) < 1e-4):
        #     breakpoint()
        last_diff = None

        if allowed_neighbors is None:
            iter_y = range(ii, len(y))
        else:
            iter_y = sorted(allowed_neighbors[i])

        for j in iter_y:
            y_val = y[j]

            # if (abs(x_val - 401.7911) < 1e-4) & (abs(y_val - 401.8036) < 1e-4):
            #     breakpoint()

            # if (abs(x_val - 0.8112) < 1e-4) & (abs(y_val - 0.8085) < 1e-4):
            #      breakpoint()
            diff = dist_fun(x_val, y_val)

            # TODO disable this for performance ...
            if last_diff is not None:
                assert diff >= last_diff
            last_diff = diff

            if diff < low_dist:
                ii = j
                continue
            if diff > high_dist:
                break

            assert diff <= high_dist and diff >= low_dist
            neighbors.add_connection(i, j)
    return neighbors


def find_neighbors_multi_vectorized(
    x: list[NDArray],
    y: list[NDArray],
    dist_funs: list[callable],
    low_dists: list[float],
    high_dists: list[float],
) -> IndexBipartite:
    """Finds Neighbors in multiple dimensions.

    This is the generalized version of `find_neighbors_vectorized`.
    This function allows for multiple dimensions to be used to find neighbors.

    It is more efficient than calling the `find_neighbors_vectorized` function
    multiple times because it does not require many intermediate representations
    of the data.

    In addition it (in theory) does some of the merging operations in a vectorized
    manner, which should be faster due to the overhead of cpu-cache moving overhead.

    Parameters
    ----------
    x, list[NDArray]:
        List of arrays to use to find neighbors.
        the two lists need to be the same length and the length of each element in
        each of the lists needs to be the same length.
        For example: if x is a list of 5 arrays, each of length 200; then y needs to
        be a list of 5 arrays, but the length of each sub-array can be any length
        (as long as they are all the same...).
    y, list[NDArray]:
        See the description of x.
    dist_funs, list[callable]:
        List of functions to calculate the distance between an element
        in `x` and an element in `y`.
        The list should be the same length as `x` and `y`.
        Note that this asumes directionality and should increase in value.

        In other words ...
        dist_fun(x[0], y[0]) < dist_fun(x[0], y[1]); assuming that y[1] > y[0]
    low_dists, list[float]:
        List of lowest value allowable as a distance for two elements
        to be considered neighbors.
        The list should be the same length as `x` and `y`.
    high_dists, list[float]:
        List of highest value allowable as a distance for two elements
        to be considered neighbors.

    Returns
    -------
    IndexBipartite:
        The neighbors found between the two arrays.
    """
    neighbors = IndexBipartite()
    final_in_range = None

    my_iter = zip(
        x,
        y,
        dist_funs,
        low_dists,
        high_dists,
    )
    for xi, yi, dist_fun, low_dist, high_dist in my_iter:
        diffs = _apply_vectorized(xi, yi, dist_fun)
        in_range = (diffs > low_dist) & (diffs < high_dist)
        if final_in_range is None:
            final_in_range = in_range
        else:
            final_in_range = final_in_range & in_range

    inds = np.where(final_in_range)
    neighbors = IndexBipartite.from_arrays(*inds)
    return neighbors


def find_neighbors_vectorized(
    x: NDArray,
    y: NDArray,
    dist_fun: callable,
    low_dist: float,
    high_dist: float,
) -> IndexBipartite:
    """Finds neighbors between to sorted arrays.

    This version uses a vectorized version of the calculation.
    In theory it should take longer, since it calculates all
    distances between elements, BUT due to vectorization, cpu caching
    and not going through the iteration in python.

    Parameters
    ----------
    x, NDArray:
        First array to use to find neighbors
    y, NDArray:
        Second array to use to find neighbors
    dist_fun:
        Function to calculate the distance between an element
        in `x` and an element in `y`.
        Note that this asumes directionality and should increase in value.
        In other words ...
        dist_fun(x[0], y[0]) < dist_fun(x[0], y[1]); assuming that y[1] > y[0]
    low_dist:
        Lowest value allowable as a distance for two elements
        to be considered neighbors
    high_dist:
        Highest value allowable as a distance for two elements
        to be considered neighbors

    Examples
    --------
    >>> x = np.array([1.,2.,3.,4.,5.,15.,25.])
    >>> y = np.array([1.1, 2.3, 3.1, 4., 25., 25.1])
    >>> dist_fun = lambda x,y: y - x
    >>> low_dist = -0.11
    >>> high_dist = 0.11
    >>> find_neighbors_vectorized(x,y,dist_fun,low_dist, high_dist)
    IndexBipartite(left_neighbors={0: [0], 2: [2], 3: [3], 6: [4, 5]},
      right_neighbors={0: [0], 2: [2], 3: [3], 4: [6], 5: [6]})
    >>> low_dist = -1.11
    >>> high_dist = -0.98
    >>> out = find_neighbors_vectorized(x,y,dist_fun,low_dist, high_dist)
    >>> out
    IndexBipartite(left_neighbors={4: [3]}, right_neighbors={3: [4]})
    >>> [[f"{x[k]} matches {y[w]}" for w in v] for k, v in out.left_neighbors.items()]
    [['5.0 matches 4.0']]
    """
    assert low_dist < high_dist
    neighbors = IndexBipartite()

    diffs = _apply_vectorized(x, y, dist_fun)
    in_range = (diffs > low_dist) & (diffs < high_dist)
    inds = np.where(in_range)
    neighbors = IndexBipartite.from_arrays(*inds)
    return neighbors


def _apply_vectorized(x: NDArray, y: NDArray, fun: callable) -> NDArray:
    """Applies a function to all combinations of elements in x and y.

    Arguments:
    ---------
    x, y: NDArray
        One dimensional Arrays to apply the function to
    fun: callable
        Function to apply to all combinations of elements in x and y

    Returns
    -------
    NDArray
        Array of shape (len(x), len(y)) with the results of the function
        applied to all combinations of elements in x and y

    Examples
    --------
    >>> x = np.array([1.,5.,15.,25.])
    >>> y = np.array([1.1, 25.1])
    >>> fun = lambda x,y: y - x
    >>> _apply_vectorized(x,y,fun)
    array([[ 0.1, 24.1],
        [ -3.9, 20.1],
        [-13.9, 10.1],
        [-23.9,  0.1]])
    """
    outs = fun(np.tile(np.expand_dims(x, axis=-1), len(y)), y)
    return outs


# @profile
def multidim_neighbor_search(
    elems1: dict[str, NDArray],
    elems2: dict[str, NDArray] | None,
    dist_ranges: dict[str, tuple[float, float]],
    dist_funs: None | dict[str, callable] = None,
    dimension_order: None | tuple[str] = None,
) -> IndexBipartite:
    """Searches for neighbors in multiple dimensions.

    Parameters
    ----------
    elems1, dict[str,NDArray]:
        Seel elems2
    elems2, dict[str,NDArray] | None:
        A dictionary of arrays.
        All arrays within one of those elements need to have the same
        length.
    dist_ranges, dict[str, tuple[float, float]]:
        maximum and minimum ranges for each of the dimensions.
    dist_funs:
        Dictionary of functions used to calculate distances.
        For details check the documentation of `find_neighbors_sorted`
    dimension_order, optional str:
        Optional tuple of strings denoting what dimensions to use.

    Examples
    --------
    >>> x1 = {"d1": np.array([1000., 1000., 2001., 3000.]),
    ...    "d2": np.array([1000., 1000.3, 2000., 3000.01])}
    >>> x2 = {"d1": np.array([1000.01, 1000.01, 2000., 3000.]),
    ...    "d2": np.array([1000.01, 1000.01, 2000., 3001.01])}
    >>> d_funs = {"d1": lambda x,y: 1e6 * (y-x)/abs(x), "d2": lambda x,y: y-x}
    >>> d_ranges = {"d1": (-10, 10), "d2": (-0.02, 0.02)}
    >>> multidim_neighbor_search(
    ...    x1, x2, d_ranges, d_funs
    ... )
    IndexBipartite(left_neighbors={0: {0, 1}, 2: {2}}, right_neighbors={0: {0}, 1: {0}, 2: {2}})
    """  # noqa: E501
    if dimension_order is None:
        dimension_order = list(elems1.keys())

    elems_1_indices = np.arange(len(elems1[dimension_order[0]]))

    if dist_funs is None:
        dist_funs = {k: lambda x, y: y - x for k in dimension_order}

    # sort all elements by their first dimension
    elems_1_order = np.argsort(elems1[dimension_order[0]])

    elems_1 = {k: v[elems_1_order] for k, v in elems1.items()}

    # The original indices are also sorted by the same dimension
    elems_1_indices = elems_1_indices[elems_1_order]

    if elems2 is not None:
        elems_2_indices = np.arange(len(elems2[dimension_order[0]]))
        elems_2_order = np.argsort(elems2[dimension_order[0]])
        elems_2 = {k: v[elems_2_order] for k, v in elems2.items()}
        elems_2_indices = elems_2_indices[elems_2_order]
    else:
        elems_2 = elems_1
        elems_2_indices = elems_1_indices

    # Set up the graph where the neighbors will be stored
    out = _multidim_neighbor_search(
        elems_1=elems_1,
        elems_2=elems_2,
        elems_1_indices=elems_1_indices,
        elems_2_indices=elems_2_indices,
        dist_ranges=dist_ranges,
        dist_funs=dist_funs,
        dimension_order=dimension_order,
    )
    return out


# @profile
def _multidim_neighbor_search(
    elems_1: dict[str, NDArray],
    elems_2: dict[str, NDArray],
    elems_1_indices: NDArray,
    elems_2_indices: NDArray,
    dist_ranges: dict[str, tuple[float, float]],
    dist_funs: dict[str, callable],
    dimension_order: tuple[str],
) -> IndexBipartite:
    """Searches for neighbors in multiple dimensions.

    This internal function is used by `multidim_neighbor_search` and
    is not intended to be used directly. Since it does not have the
    safety guarantees that the public function has.

    For the parameters deltails please check the documentation of the
    public function.
    """
    neighbors = IndexBipartite()
    # ii = 0
    assert is_sorted(elems_1[dimension_order[0]])

    for i in range(len(elems_1[dimension_order[0]])):
        # Allowable indices is a list that maps the indices that
        # are still viable to use, mapping to the original indices
        allowable_indices = None
        for dimension in dimension_order:
            curr_dimension_matches = []
            dist_fun = dist_funs[dimension]
            low_dist, high_dist = dist_ranges[dimension]
            x = elems_1[dimension]
            y = elems_2[dimension]

            x_val = x[i]

            assert low_dist < high_dist
            assert dist_fun(low_dist, high_dist) > 0
            assert dist_fun(high_dist, low_dist) < 0

            # we generate an iterable that yields the indices of
            # the original array in increasing order of the current
            # dimension
            if allowable_indices is not None:
                if len(allowable_indices) == 0:
                    break
                # we can only search in the allowable indices
                # from the previous dimension
                allowed_y = y[allowable_indices]

                match_indices = allowed_y >= x_val + low_dist
                match_indices = match_indices & (allowed_y <= x_val + high_dist)
                curr_dimension_matches = allowable_indices[match_indices]
            else:
                curr_dimension_matches = np.arange(
                    np.searchsorted(y, x_val + low_dist),
                    np.searchsorted(y, x_val + high_dist),
                )

                # Speed test
                match_indices = y >= x_val + low_dist

                # Surprisingly, this is slower ...
                # ii = np.searchsorted(y[ii:], x_val + low_dist) + ii
                # oi = np.searchsorted(y[ii:], x_val + high_dist) + ii
                # curr_dimension_matches_o = np.arange(ii, oi)
                # assert np.all(curr_dimension_matches == curr_dimension_matches_o)

            # if allowable_indices is not None:
            #     assert all(x in allowable_indices for x in curr_dimension_matches)
            allowable_indices = curr_dimension_matches

        # After going though all dimensions, we have the allowable indices
        # for the current element in the first array
        for j in curr_dimension_matches:
            neighbors.add_connection(i, j)

    # Now we have to map the indices back to the original indices
    # TODO check if vectorizing this is faster
    neighbors = IndexBipartite(
        left_neighbors={
            elems_1_indices[k]: {elems_2_indices[w] for w in v}
            for k, v in neighbors.left_neighbors.items()
        },
        right_neighbors={
            elems_2_indices[k]: {elems_1_indices[w] for w in v}
            for k, v in neighbors.right_neighbors.items()
        },
    )
    return neighbors
