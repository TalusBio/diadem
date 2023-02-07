from diadem.index.protein_index import ProteinNGram


def test_ngram_works(shared_datadir):
    """Tests that the protein n-gram can be generated from a fasta file.

    It also tests that the right protein is returned from a random peptide.
    """
    query_peptide = "NDGATILSLLDVQHPAGK"
    out = ["P12612"]
    ngram = ProteinNGram.from_fasta(str(shared_datadir / "small-yeast.fasta"))

    assert ngram.search_ngram(query_peptide) == out
