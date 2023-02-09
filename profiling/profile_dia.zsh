
python -m pip install ../.
DEBUG_DIADEM=1 python -m cProfile -s tottime -m diadem.cli search \
    --mzml_file profiling_data/Chessfest_Plate3_RH4_DMSO_DIA.mzML \
    --fasta profiling_data/uniprot_human_sp_canonical_2021-11-19_crap.fasta \
    --out_prefix "" --mode DIA > profile.txt
