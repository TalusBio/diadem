

python -m cProfile -s tottime -m diadem.cli search \
    --mzml_file ../benchmarking_tests/Chessfest_Plate3_RH4_DMSO_DIA.mzML \
    --fasta ../benchmarking_tests/uniprot_human_sp_canonical_2021-11-19_crap.fasta \
    --out_prefix "" --mode DIA > profile.txt
