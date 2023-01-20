"""Initializes DIAdem!."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        pass

except ImportError:
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        pass
