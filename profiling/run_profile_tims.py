from diadem import cli
from diadem.config import DiademConfig

cli.setup_logger()
cli.diadem_main(
    fasta_path="./profiling_data/UP000005640_9606.fasta",
    data_path="./profiling_data/20210510_TIMS03_EVO03_PaSk_SA_HeLa_50ng_5_6min_DIA_high_speed_S1-B2_1_25186.hdf",
    config=DiademConfig(
        run_parallelism=1,  # Needs to use 1 core for profiling
        run_max_peaks=20000,
        run_allowed_fails=500,
        g_tolerances=(25, 25),
        g_tolerance_units=("ppm", "ppm"),
        g_ims_tolerance=0.03,
        g_ims_tolerance_unit="abs",
    ),
    out_prefix="lineprofile_results_tims/results",
)
