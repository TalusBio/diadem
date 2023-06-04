"""Verify that our interface base class works."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from diadem.interfaces import BaseDiademInterface, RequiredColumn


def test_required_column():
    """Test the required column dataclass."""
    cols = [("foo", pl.datatypes.Float32), ("bar", pl.datatypes.Field)]
    req = RequiredColumn.from_iter(cols)

    for col, result in zip(cols, req):
        assert col[0] == result.name
        assert col[1] == result.dtype


def test_base_interface_init():
    """Test our initialization."""

    class RealInterface(BaseDiademInterface):
        """A dummy interface."""

        def __init__(self, data):
            """Init the thing."""
            super().__init__(data)

        @property
        def schema(self):
            """The required columns."""
            return [
                RequiredColumn("foo", pl.datatypes.Float32),
                RequiredColumn("bar", pl.datatypes.Utf8),
            ]

    good_df = pl.DataFrame({"foo": [1.0, 2.0], "bar": ["a", "b"]})
    interface = RealInterface(good_df)
    assert_frame_equal(interface.data.collect(), good_df)

    bad_df = pl.DataFrame({"foo": ["a", "b"], "bar": ["a", "b"]})
    with pytest.raises(ValueError) as err:
        RealInterface(bad_df)

    assert "wrong data type" in str(err.value)

    bad_df = pl.DataFrame({"bar": ["a", "b"]})
    with pytest.raises(ValueError) as err:
        RealInterface(bad_df)

    assert "missing" in str(err.value)


def test_base_interface_from_parquet(tmp_path):
    """Test loading a parquet file."""

    class RealInterface(BaseDiademInterface):
        """A dummy interface."""

        def __init__(self, data):
            """Init the thing."""
            super().__init__(data)

        @property
        def schema(self):
            """The required columns."""
            return [
                RequiredColumn("foo", pl.datatypes.Float32),
                RequiredColumn("bar", pl.datatypes.Utf8),
            ]

    good_df = pl.DataFrame({"foo": [1.0, 2.0], "bar": ["a", "b"]})
    good_df.write_parquet(tmp_path / "test.parquet")
    interface = RealInterface.from_parquet(tmp_path / "test.parquet")
    pl.testing.assert_frame_equal(interface.data.collect(), good_df)
