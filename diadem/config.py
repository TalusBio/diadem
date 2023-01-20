from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from typing import Literal

import tomli_w
from loguru._logger import Logger

if sys.version_info <= (3, 11):
    import tomli as tomllib
else:
    import tomllib

from ms2ml import Config

ScoringFunctions = Literal["Hyperscore"]
ScoreReference = Literal["Center", "Correlation", "ScaledCorrelation"]
MassError = Literal["da", "ppm"]

# TODO add chunksize to config ... and use it ...


@dataclass(frozen=True, eq=True)
class DiademConfig:  # noqa
    g_tolerances: tuple[float, ...] = field(default=(20, 10))
    g_tolerance_units: tuple[MassError, ...] = field(default=("ppm", "ppm"))

    peptide_length_range: tuple[int, int] = field(
        default=(7, 25),
    )
    peptide_mz_range: tuple[float, float] = field(
        default=(399.0, 1001.0),
    )

    precursor_charges: tuple[float, float] = field(
        default=(2, 3),
    )

    ion_series: str = field(
        default="by",
    )
    ion_charges: tuple[int, ...] = field(
        default=(1,),
    )
    ion_mz_range: tuple[float, float] = field(
        default=(250, 2000.0),
    )

    db_enzyme: str = field(default="trypsin")
    db_max_missed_cleavages: int = 2
    db_bucket_size: int = 2**15

    # Variable mods
    # Static mods

    # Main score
    # Currently unused ...
    scoring_score_function: ScoringFunctions = "Hyperscore"

    run_max_peaks: int = 1e6

    # the 5k number comes from the neviskii lab paper on deisotoping
    run_max_peaks_per_spec: int = 5_000
    run_deconvolute_spectra: bool = True
    run_min_peak_intensity: float = 100
    # run_debug_log_frequency: int = 200
    run_debug_log_frequency: int = 20
    run_allowed_fails: int = 50
    run_window_size: int = 21

    # Min intensity to consider for matching and extracting
    run_min_intensity_ratio: float = 0.011
    run_min_correlation_score: float = 0.25

    run_scaling_ratio = 0.01
    run_scalin_limits: tuple[float, float] = (0.01, 0.999)

    @property
    def ms2ml_config(self) -> Config:
        """Returns the ms2ml config.

        It exports all the parameters that are used inside of
        ms2ml to its own configuration object.
        """
        conf = Config(
            g_tolerances=self.g_tolerances,
            g_tolerance_units=self.g_tolerance_units,
            peptide_length_range=self.peptide_length_range,
            precursor_charges=self.precursor_charges,
            ion_series=self.ion_series,
            ion_charges=self.ion_charges,
            peptide_mz_range=self.peptide_mz_range,
        )
        return conf

    def log(self, logger: Logger, level: str = "INFO") -> None:
        """Logs all the configurations using the passed logger."""
        logger.log(level, "Diadem Configuration:")
        for k, v in asdict(self).items():
            logger.log(level, f"{k}: {v}")

    def to_toml(self, path: str) -> None:
        """Writes the config to a toml file."""
        out = {}
        for k, v in asdict(self).items():
            if isinstance(v, tuple):
                v = [x if x is not None else "__NONE__" for x in v]
            elif v is None:
                v = "__NONE__"
            out[k] = v
        with open(path, "wb") as f:
            tomli_w.dump(out, f)

    @classmethod
    def from_toml(cls, path: str) -> DiademConfig:
        """Loads a config from a toml file."""
        with open(path, "rb") as f:
            config = tomllib.load(f)

        out_config = {}
        for k, v in config.items():
            if isinstance(v, list):
                v = tuple(x if x != "__NONE__" else None for x in v)
            elif v == "__NONE__":
                v = None
            out_config[k] = v

        return cls(**out_config)

    @classmethod
    def from_args(cls, args: Namespace) -> DiademConfig:
        """Simple wrapper to start a config using parsed arguments.

        The passed arguments must contain a `config` attribute,
        which is a path-like string that points to a toml configuration
        file.
        """
        return cls.from_toml(args.config)
