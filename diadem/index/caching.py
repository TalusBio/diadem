from __future__ import annotations

import hashlib
from pathlib import Path

from platformdirs import PlatformDirs

import diadem


def get_file_md5(file: Path | str) -> str:
    """Calculates the md5 of a file and returns a printable string representing it."""
    with open(file, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def file_cache_dir(file: Path | str) -> Path:
    """Returns the caching directory for a file.

    This directory is built using the md5 digest
    and located inside the cache directory of the user.
    """
    md5 = get_file_md5(file)
    dirs = PlatformDirs("diadem", "talusbio", version=diadem.__version__)
    return Path(dirs.user_cache_dir) / md5
