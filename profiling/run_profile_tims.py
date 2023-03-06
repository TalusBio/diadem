from diadem import cli
from diadem.config import DiademConfig

cli.setup_logger()
cli.diadem_main(
    fasta_path="./profiling_data/UP000000625_83333.fasta",
    data_path="./profiling_data/LFQ_timsTOFPro_diaPASEF_Ecoli_01.d",
    config=DiademConfig(run_parallelism=1),
    out_prefix="lineprofile_results_tims/results",
)
