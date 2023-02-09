from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from diadem.utils import get_slice_inds, is_sorted

SortingLevel = Literal["ms1", "ms2"]


@dataclass
class FragmentBucket:
    """A bucket of fragments.

    This dataclass is used to store fragments and precursor mzs, in a way
    that it relates to the original data.

    The dataclass includes methods to filter and sort the bucket.

    Parameters
    ----------
    fragment_mzs : NDArray[np.float32]
        The fragment m/z values.
    fragment_series : NDArray[np.str]
        The fragment series (b, y, etc).
    precursor_ids : NDArray[np.int64]
        The precursor ids.
    precursor_mzs : NDArray[np.float32]
        The precursor m/z values.
    sorting_level : SortingLevel, optional
        The level to sort the bucket by, by default "ms1"
    is_sorted : bool, optional
        Whether the bucket is already sorted, by default False
        It will not be sorted or checked for sorting if True is passed.

    All array arguments need to be the same length.

    """

    fragment_mzs: NDArray[np.float32]
    fragment_series: NDArray[np.str]
    precursor_ids: NDArray[np.int64]
    precursor_mzs: NDArray[np.float32]
    sorting_level: SortingLevel = "ms1"
    is_sorted: bool = False

    def __post_init__(self) -> None:
        """Sorts the fragments if needed after initialization."""
        # assert len(
        # set(len(x) for x in
        # [self.precursor_ids, self.precursor_mzs,
        # self.fragment_mzs, self.fragment_series])) == 1

        if not self.is_sorted:
            self.sort()
        self.is_sorted = True

    def sort(self, level: SortingLevel | None = None) -> None:
        """Sorts the fragments by the given (ms)level."""
        if level is not None:
            self.sorting_level = level

        if self.sorting_level is not None and not is_sorted(self.sorting_level_array):
            idx = np.argsort(self.sorting_level_array)
            self.fragment_mzs = self.fragment_mzs[idx]
            self.fragment_series = self.fragment_series[idx]
            self.precursor_ids = self.precursor_ids[idx]
            self.precursor_mzs = self.precursor_mzs[idx]

    @property
    def sorting_level_array(self) -> NDArray:
        """Returns the array that is used for sorting.

        This is either the precursor mzs or the fragment mzs depending on the
        sorting level.
        """
        if self.sorting_level == "ms1":
            return self.precursor_mzs
        elif self.sorting_level == "ms2":
            return self.fragment_mzs
        else:
            raise ValueError(f"Unknown sorting level: {self.sorting_level}")

    @property
    def min_frag_mz(self) -> float:
        """Returns the minimum fragment mz in the bucket."""
        if not hasattr(self, "_min_frag_mz"):
            self._min_frag_mz = np.min(self.fragment_mzs)

        return self._min_frag_mz

    def __len__(self) -> int:
        """Returns the number of fragments in the bucket."""
        return len(self.fragment_mzs)

    def filter_ms1(self, min_mz: float, max_mz: float) -> FragmentBucket:
        """Filters the bucket by the given precursor m/z range."""
        if self.sorting_level == "ms1" and self.is_sorted:
            slc = get_slice_inds(self.precursor_mzs, minval=min_mz, maxval=max_mz)
        else:
            slc = (self.precursor_mzs >= min_mz) * (self.precursor_mzs <= max_mz)
        out = FragmentBucket(
            fragment_mzs=self.fragment_mzs[slc],
            fragment_series=self.fragment_series[slc],
            precursor_ids=self.precursor_ids[slc],
            precursor_mzs=self.precursor_mzs[slc],
            sorting_level=self.sorting_level,
            is_sorted=self.is_sorted,
        )
        return out

    def filter_ms2(self, min_mz: float, max_mz: float) -> FragmentBucket:
        """Filters the bucket by the given fragment m/z range."""
        if self.sorting_level == "ms2" and self.is_sorted:
            slc = get_slice_inds(self.fragment_mzs, minval=min_mz, maxval=max_mz)
        else:
            slc = (self.fragment_mzs >= min_mz) * (self.fragment_mzs <= max_mz)

        out = FragmentBucket(
            fragment_mzs=self.fragment_mzs[slc],
            fragment_series=self.fragment_series[slc],
            precursor_ids=self.precursor_ids[slc],
            precursor_mzs=self.precursor_mzs[slc],
            sorting_level=self.sorting_level,
            is_sorted=self.is_sorted,
        )
        return out

    @classmethod
    def concatenate(cls, *args: FragmentBucket) -> FragmentBucket:
        """Concatenates the given buckets.

        The sorting level of the buckets must be the same.

        Parameters
        ----------
        *args : list[FragmentBucket]
            An arbitrary number of buckets to concatenate

        Usage
        -----
        > bucket1 = FragmentBucket(...)
        > bucket2 = FragmentBucket(...)
        > bucket3 = FragmentBucket(...)
        > bucket = FragmentBucket.concatenate(bucket1, bucket2, bucket3)
        """
        sorting_level = {x.sorting_level for x in args}
        assert (
            len(sorting_level) == 1
        ), "Cannot concatenate buckets with different sorting levels"
        return cls(
            fragment_mzs=np.concatenate([x.fragment_mzs for x in args]),
            fragment_series=np.concatenate([x.fragment_series for x in args]),
            precursor_ids=np.concatenate([x.precursor_ids for x in args]),
            precursor_mzs=np.concatenate([x.precursor_mzs for x in args]),
            is_sorted=True,
            sorting_level=sorting_level.pop(),
        )


@dataclass
class FragmentBucketList:
    """A list of fragment buckets.

    This class implements many methods to manipulate and query lists of
    fragment buckets (sorting, splitting, filtering, querying).

    Check the documentation of the individual methods for more information.

    Parameters
    ----------
    buckets : list[FragmentBucket]
        The list of buckets.

    """

    buckets: list[FragmentBucket]

    def prefilter_ms1(
        self,
        min_mz: float,
        max_mz: float,
        num_decimals: int = 3,
        max_frag_mz: float = 2000,
        progress: bool = False,
    ) -> PrefilteredMS1BucketList:
        """Prefilters the buckets by the given precursor m/z range.

        This method is used to prefilter the buckets by the precursor m/z
        range. This is useful when the precursor m/z range is known in
        advance for a large number of fragments (DIA data)
        """
        out = [x.filter_ms1(min_mz=min_mz, max_mz=max_mz) for x in self.buckets]
        out = [x for x in out if x if len(x)]
        out = PrefilteredMS1BucketList(
            buckets=out,
            num_decimal=num_decimals,
            max_frag_mz=max_frag_mz,
            progress=progress,
        )
        return out

    def buckets_matching_ms2(self, min_mz: float, max_mz: float) -> FragmentBucketList:
        """Returns the buckets that contain fragments in the given m/z range.

        Note: the fragments are not filtered internally, so not all of the fragments in
        all of the buckets will match the mz range (but at least one will).
        """
        ms2_slc = get_slice_inds(self.bucket_mins, min_mz, max_mz)
        return self[ms2_slc]

    @property
    def bucket_mins(self) -> NDArray[np.float32]:
        """Returns the minimum fragment m/z for each bucket."""
        if not hasattr(self, "_bucket_mins"):
            self._bucket_mins = np.array([x.min_frag_mz for x in self.buckets])

        return self._bucket_mins

    @property
    def min_ms2_mz(self) -> float:
        """Returns the minimum fragment m/z across all buckets."""
        return np.min(self.bucket_mins)

    def __getitem__(self, val: int | slice) -> FragmentBucket | FragmentBucketList:
        """Subsets the list of buckets to either get a single one or a range."""
        if isinstance(val, slice):
            return FragmentBucketList(self.buckets[val])
        elif isinstance(val, int):
            return self.buckets[val]
        else:
            raise ValueError(
                f"Subsetting FragmentBucketList with {type(val)}: {val} is not"
                " supported"
            )

    @classmethod
    def from_arrays(
        cls,
        fragment_mzs: NDArray[np.float32],
        fragment_series: NDArray[np.str],
        precursor_ids: NDArray[np.int64],
        precursor_mzs: NDArray[np.float32],
        chunksize: int,
        sorting_level: str,
        been_sorted: bool = False,
    ) -> FragmentBucketList:
        """Creates a FragmentBucketList from the given arrays.

        Parameters
        ----------
        fragment_mzs : NDArray[np.float32]
            An array of fragment m/z values.
        fragment_series : NDArray[np.str]
            An array of fragment ion series.
        precursor_ids : NDArray[np.int64]
            An array of sequence ids. (unique identifier of a peptide sequence)
        precursor_mzs : NDArray[np.float32]
            An array of precursor m/z values.
        chunksize : int
            The size of the chunks to split the arrays into.
        sorting_level : str
            The sorting level of the arrays (ms1/ms2).
        been_sorted : bool, optional
            Whether the arrays have already been sorted, by default False

        """
        if not is_sorted(fragment_mzs):
            raise ValueError("fragment_mzs must be sorted")
        buckets = []

        vals = list(range(0, len(fragment_mzs), chunksize))
        for i in tqdm(vals, desc="Fragment Buckets"):
            buckets.append(
                FragmentBucket(
                    fragment_mzs=fragment_mzs[i : i + chunksize],
                    fragment_series=fragment_series[i : i + chunksize],
                    precursor_ids=precursor_ids[i : i + chunksize],
                    precursor_mzs=precursor_mzs[i : i + chunksize],
                    sorting_level=sorting_level,
                    is_sorted=been_sorted,
                )
            )
        return cls(buckets)

    def sort(self, level: SortingLevel) -> None:
        """Sorts the buckets by the given level."""
        for b in self.buckets:
            b.sort(level)

    # @profile
    def yield_candidates(
        self, ms2_range: tuple[float, float], ms1_range: tuple[float, float]
    ) -> None | Iterator[tuple[int, float, str]]:
        """Yields fragments that match the passed masses.

        Parameters
        ----------
        ms2_range : tuple[float, float]
            The minimum and maximum m/z of the fragment to search for.
        ms1_range : tuple[float, float]
            The minimum and maximum m/z of the precursor to search for.


        Yields
        ------
        int : id of the peptide
        float : mz of the fragment
        str : series the fragment belongs to (y, b, etc.)

        """
        if self.min_ms2_mz > ms2_range[1]:
            # This means the queried fragment is smaller than the
            # smallest in the database
            return None

        ms2_match_buckets = self.buckets_matching_ms2(ms2_range[0], ms2_range[1])

        for bucket in ms2_match_buckets.buckets:
            bucket = bucket.filter_ms1(ms1_range[0], ms1_range[1])
            if len(bucket) == 0:
                continue

            yield from zip(
                bucket.precursor_ids, bucket.fragment_mzs, bucket.fragment_series
            )


class PrefilteredMS1BucketList:
    """Variant of the BucketList object that contains only spectra in a mz1 range.

    Since it is already filtered by ms1, the search for peaks is a lot faster.
    """

    def __init__(
        self,
        buckets: list[FragmentBucket],
        num_decimal: int = 3,
        max_frag_mz: float = 2000.0,
        progress: bool = False,
    ) -> None:
        """Initializes the PrefilteredMS1BucketList object.

        Parameters
        ----------
        buckets : list[FragmentBucket]
            The list of buckets to initialize the object with.
            (will be filtered after initializing)
        num_decimal : int, optional
            The number of decimal places to round the m/z values to, by default 3
        max_frag_mz : float, optional
            The maximum fragment m/z to consider, by default 2000.0
        progress : bool, optional
            Whether to show a progress bar, by default False
        """
        self.prod_num = 10**num_decimal
        self.buckets = [[] for _ in range(int(max_frag_mz * self.prod_num))]
        iterator = tqdm(buckets, desc="Unpacking buckets", disable=not progress)
        for bucket in iterator:
            unpacked = self.unpack_bucket(bucket)
            for k, v in unpacked.items():
                self.buckets[k].append(v)

        iterator = enumerate(tqdm(self.buckets, disable=not progress))
        min_ms2_mz = 2**15
        for i, e in iterator:
            if e:
                cat_buckets = FragmentBucket.concatenate(*e)
                cat_buckets.sorting_level = "ms2"
                cat_buckets.sort()
                if len(cat_buckets) > 0:
                    self.buckets[i] = cat_buckets
                    if self.buckets[i].min_frag_mz < min_ms2_mz:
                        min_ms2_mz = self.buckets[i].min_frag_mz

        self.min_ms2_mz = min_ms2_mz

    def unpack_bucket(self, bucket: FragmentBucket) -> dict[int, FragmentBucket]:
        """Unpacks a bucket into a dictionary of buckets.

        For the unpacking, the bucket is split by the fragment mz
        using a precision of x decimal places, that is defined by the
        prod_num attribute.
        """
        # TODO decide if this should be a fragment bucket method...
        integerized = (bucket.fragment_mzs * self.prod_num).astype(int)
        uniqs = np.unique(integerized)
        out = {}

        for u in uniqs:
            idxs = integerized == u

            # TODO consider here not traking precursor mzs anymore
            out[u] = FragmentBucket(
                fragment_mzs=bucket.fragment_mzs[idxs],
                fragment_series=bucket.fragment_series[idxs],
                precursor_ids=bucket.precursor_ids[idxs],
                precursor_mzs=bucket.precursor_mzs[idxs],
            )
        return out

    def yield_buckets_matching_ms2(
        self, min_mz: float, max_mz: float
    ) -> Iterator[FragmentBucket]:
        """Yields buckets that match the passed ms2 range."""
        min_index = max(0, int(min_mz * self.prod_num) - 1)
        max_index = int(max_mz * self.prod_num) + 1

        for i in range(min_index, max_index):
            if len(self.buckets[i]):
                yield self.buckets[i]

    # @profile
    def yield_candidates(
        self, ms2_range: tuple[float, float], ms1_range: tuple[float, float]
    ) -> Iterator[tuple[int, float, str]]:
        """Yields fragments that match the passed masses.

        Parameters
        ----------
        ms2_range : tuple[float, float]
            The minimum and maximum m/z of the fragment to search for.
        ms1_range : tuple[float, float]
            The minimum and maximum m/z of the precursor to search for.
            This argument is not used in this database, since the whole
            database has already been filtered for ms1.


        Yields
        ------
        int : id of the peptide
        float : mz of the fragment
        str : series the fragment belongs to (y, b, etc.)

        """
        # Note: the ms1 range argument is not used here but kept
        # to maintain the same interface as the non-prefiltered database.
        min_mz, max_mz = ms2_range
        # last_mz = 0
        for x in self.yield_buckets_matching_ms2(min_mz, max_mz):
            for sid, fragmz, fragseries in zip(
                x.precursor_ids, x.fragment_mzs, x.fragment_series
            ):
                if fragmz > max_mz:
                    break
                # assert fragmz >= last_mz, "Fragment mzs not sorted"
                # last_mz = fragmz
                yield sid, fragmz, fragseries
