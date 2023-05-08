"""Interfaces for results from Diadem modules.

This module provides a base class for defining interfaces between
Diadem modules. Each child class should provides access to dataframes
representing results from a Diadem function/method, optionally backed
by a parquet file. The child class can then be used by other Diadem
modules to perform the next step of an algorithm.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from os import PathLike

import polars as pl

POLY_DTYPES = (
    pl.datatypes.FLOAT_DTYPES,
    pl.datatypes.INTEGER_DTYPES,
    pl.datatypes.UNSIGNED_INTEGER_DTYPES,
)


class BaseDiademInterface(ABC):
    """A base class for interfaces in Diadem.

    Parameters
    ----------
    data : DataFrame | LazyFrame
        A polars or pandas DataFrame containing the data for the interface.
    """

    @classmethod
    def from_parquet(cls, source: PathLike) -> None:
        """Initialize the interface from a parquet file.

        Parameters
        ----------
        source : PathLike
            The parquet file.
        """
        return cls(pl.scan_parquet(source))

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        """Initialize the interface."""
        try:
            self.data = data.lazy()
        except AttributeError:
            self.data = pl.from_pandas(data).lazy()

        self.validate_columns()

    def validate_columns(self) -> None:
        """Verify that the required columns are present and the correct dtype."""
        req_dtypes = {
            c.name: _check_for_poly_dtype(c.dtype) for c in self.required_columns
        }

        dtype_errors = []
        missing_columns = set(req_dtypes.keys())
        for col, dtype in zip(self.data.columns, self.data.dtypes):
            dtype = _check_for_poly_dtype(dtype)
            if not dtype == req_dtypes[dtype]:
                dtype_errors.append((col, dtype, req_dtypes[dtype]))

            try:
                missing_columns.remove(col)
            except KeyError:
                pass

        if not dtype_errors and not missing_columns:
            return

        dtype_msg = [
            f"  - {n}: {d} (Expected {'or'.join(r)})" for n, d, r in dtype_errors
        ]
        missing_msg = [f"  - {c}" for c in missing_columns]

        msg = []
        if dtype_errors:
            msg.append("Some columns were of the wrong data type:")
            msg += dtype_msg

        if missing_columns:
            msg.append("Some columns were missing:")
            msg += missing_msg

        raise ValueError("\n".join(msg))

    @abstractmethod
    @property
    def required_columns(self) -> Iterable[RequiredColumn]:
        """The required columns for the underlying DataFrame."""


@dataclass
class RequiredColumn:
    """Specify a required column.

    Parameters
    ----------
    name : str
        The column name.
    dtype : pl.datatypes.Datatype
        The polars data type for the column.
    """

    name: str
    dtype: pl.datatypes.DataType

    @classmethod
    def from_iter(
        cls,
        columns: Iterable[tuple[str, pl.DataType], ...],
    ) -> Generator[RequiredColumn, ...]:
        """Create required columns from an iterable.

        Parameters
        ----------
        columns : Iterable[tuple[str, pl.DataType]]
            2-tuples of name-dtype pairs to be required.

        Yields
        ------
        RequiredColumn
        """
        for col, dtype in columns:
            yield cls(col, dtype)


def _check_for_poly_dtype(
    dtype: pl.datatypes.DataType,
) -> set[pl.datatypes.DataType, ...]:
    """Check for poly-dtypes, like floats and ints.

    Parameters
    ----------
    dtype : pl.datatypes.DataType
        A polars datatype

    Returns
    -------
    set[pl.datatypes.DataType]
    """
    dtype = set(dtype)
    for poly_dtype in POLY_DTYPES:
        if dtype.issubset(poly_dtype):
            return poly_dtype

    return dtype
