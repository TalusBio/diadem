from __future__ import annotations

import re
from collections import defaultdict
from os import PathLike

from pyteomics.fasta import FASTA
from tqdm.auto import tqdm

FASTA_NAME_REGEX = re.compile(r"^.*\|(.*)\|.*$")


class ProteinNGram:
    """Implements an n-gram to fast lookup of proteins that match a peptide.

    Examples
    --------
    >>> base_ngram = {'AA': {1,3}, 'AB': {2,3}, 'AC': {1, 4}, 'CA': {4}}
    >>> inv_index = {1: "Prot1", 2: "Prot2", 3: "Prot3", 4: "Prot4"}
    >>> ngram = ProteinNGram(ngram = base_ngram, inv_alias = inv_index)
    >>> ngram.search_ngram("AAC")
    ['Prot1']
    >>> ngram.search_ngram("CAC")
    ['Prot4']
    """

    __slots__ = ("ngram_size", "ngram", "inv_alias")

    def __init__(self, ngram: dict[str, set[int]], inv_alias: dict[int, str]) -> None:
        """Initialized an ngram for fast lookup.

        For details check the main class docstring.
        """
        keys = list(ngram)
        if not all(len(keys[0]) == len(k) for k in ngram):
            raise ValueError("All ngram keys need to be the same length")
        self.ngram_size: int = len(keys[0])
        self.ngram = ngram
        self.inv_alias = inv_alias

    def search_ngram(self, entry: str) -> list[str]:
        """Searches a sequence using the n-gram and returns the matches."""
        candidates = None
        for x in [
            entry[x : x + self.ngram_size]
            for x in range(1 + len(entry) - self.ngram_size)
        ]:
            if len(x) < self.ngram_size:
                break

            if candidates is None:
                candidates = self.ngram[x]
            else:
                candidates = candidates.intersection(self.ngram[x])
                if len(candidates) == 1:
                    break

        if candidates is None:
            raise ValueError(f"No candidates found for {entry} in the n-gram database")
        out = [self.inv_alias[x] for x in candidates]
        return out

    @staticmethod
    def from_fasta(
        fasta_file: PathLike | str, ngram_size: int = 4, progress: bool = True
    ) -> ProteinNGram:
        """Builds a protein n-gram from a fasta file.

        Parameters
        ----------
        fasta_file:
            Path-like or string representing the fasta file to read in order
            to build the index.
        ngram_size:
            Size of the chunks that will be used to build the n-gram, should
            be smaller than the smallest peptide to be searched. Longer sequences
            should give a more unique aspect to it but a larger index is built.
        progress:
            Whether to show a progress bar while building the index.


        """
        ngram = defaultdict(set)
        inv_alias = {}

        for i, entry in tqdm(
            enumerate(FASTA(fasta_file)),
            disable=not progress,
            desc="Building peptide n-gram index",
        ):
            entry_name = FASTA_NAME_REGEX.search(entry.description).group(1)
            sequence = entry.sequence

            inv_alias[i] = entry_name
            for x in [
                sequence[x : x + ngram_size]
                for x in range(1 + len(sequence) - ngram_size)
            ]:
                if len(x) < ngram_size:
                    break
                ngram[x].add(i)

        return ProteinNGram(ngram=ngram, inv_alias=inv_alias)
