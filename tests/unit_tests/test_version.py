"""Test that setuptools-scm is working correctly"""
import diadem


def test_version():
    """Check that the version is not None"""
    assert diadem.__version__ is not None
