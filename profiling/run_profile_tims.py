from diadem import cli
from diadem.config import DiademConfig

cli.setup_logger()
cli.diadem_main(
    fasta_path="./profiling_data/UP000000625_83333.fasta",
    data_path="./profiling_data/LFQ_timsTOFPro_diaPASEF_Ecoli_01.hdf",
    config=DiademConfig(
        run_parallelism=3,
        run_max_peaks=20000,
        run_allowed_fails=500,
        g_tolerances=(25, 25),
        g_tolerance_units=("ppm", "ppm"),
    ),
    out_prefix="lineprofile_results_tims/results",
)
