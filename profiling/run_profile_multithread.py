from diadem import cli
from diadem.config import DiademConfig

cli.setup_logger(level="INFO")
cli.diadem_main(
    fasta_path="./profiling_data/uniprot_human_sp_canonical_2021-11-19_crap.fasta",
    data_path="./profiling_data/Chessfest_Plate3_RH4_DMSO_DIA.mzML",
    config=DiademConfig(run_parallelism=-4),
    out_prefix="lineprofile_results_multithread/results",
)
