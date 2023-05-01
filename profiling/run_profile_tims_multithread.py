from diadem import cli
from diadem.config import DiademConfig

cli.setup_logger()
cli.diadem_main(
    fasta_path="./profiling_data/UP000005640_9606.fasta",
    # data_path="./profiling_data/20210510_TIMS03_EVO03_PaSk_SA_HeLa_50ng_5_6min_DIA_high_speed_S1-B2_1_25186.hdf", # noqa: E501
    data_path="./profiling_data/Hela_25ng_22min_6x3_short_S2-A4_1_118.hdf",
    config=DiademConfig(
        run_parallelism=4,  # Needs to use 1 core for profiling
        run_max_peaks=20000,
        run_allowed_fails=5000,
        g_tolerances=(0.02, 0.02),
        g_tolerance_units=("da", "da"),
        g_ims_tolerance=0.02,
        g_ims_tolerance_unit="abs",
        run_min_correlation_score=0.2,
        run_min_intensity_ratio=0.01,
        peptide_mz_range=(400, 2000),
        run_scaling_ratio=0.001,
        run_scalin_limits=(0.01, 0.99),
    ),
    out_prefix="lineprofile_results_tims_mt/results",
)
