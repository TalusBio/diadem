import random
from random import choices, randint, sample

import numpy as np
import pytest
from tqdm.auto import tqdm

aa_counts = {
    "L": 1133662,
    "S": 947813,
    "E": 808465,
    "A": 797562,
    "G": 747576,
    "P": 718408,
    "V": 678489,
    "K": 652419,
    "R": 641184,
    "T": 608864,
    "Q": 542455,
    "D": 539184,
    "I": 493546,
    "F": 415152,
    "N": 408255,
    "Y": 302891,
    "H": 298246,
    "C": 261599,
    "M": 242596,
    "W": 138148,
    "U": 36,
}


def make_protein(prot_length: int) -> str:
    """Makes the sequence of a fake protein of the passed length."""
    return "".join(
        sample(list(aa_counts.keys()), counts=list(aa_counts.values()), k=prot_length)
    )


def make_fakename():
    """Makes a fake fasta header with a random name."""
    return ">" + "".join(choices(list(aa_counts), k=25))


def make_fasta(outfile, n=5_000):
    """Makes a fake fasta file with n number of proteins."""
    random.seed(42)
    np.random.seed(42)
    with open(outfile, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n)):
            prot_length = randint(100, 1000)
            prot_seq = make_protein(prot_length=prot_length) + "\n"
            prot_name = make_fakename() + "\n"
            f.writelines([prot_name, prot_seq])


def fake_spectra_tuple(rng: np.random.Generator, npeaks):
    """Makes random arrays that mimic spectra."""
    mzs = rng.uniform(150, 1900, size=npeaks)
    precursor_mz = rng.uniform(150, 1900, size=1)
    ints = rng.exponential(scale=10_000, size=npeaks)

    return mzs, ints, precursor_mz


@pytest.fixture
def fake_spectra_tuples_100():
    """Fixture that generate 100 fake spectra for testing."""
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    specs = [fake_spectra_tuple(rng, npeaks=500) for _ in range(100)]
    return specs


@pytest.fixture
def fake_5k_fasta(tmpdir):
    """Fixture that generates a fasta file with 5k fake proteins."""
    fasta_location = tmpdir / "myfakefasta.fasta"
    make_fasta(outfile=fasta_location, n=5000)
    return fasta_location
