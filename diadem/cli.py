from __future__ import annotations

import os
import sys

import rich_click as click
from loguru import logger

from diadem.config import DiademConfig
from diadem.search.dda import dda_main
from diadem.search.diadem import diadem_main


def setup_logger(logger=logger, level="INFO") -> None:
    """Sets up the logger to level info and a sink to a log file."""
    if "DEBUG_DIADEM" in os.environ:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.add("diadem_log.log", diagnose=False)
        logger.error("RUNNING DIADEM IN DEBUG MODE")
    else:
        logger.remove()
        logger.add(sys.stderr, level=level)
        logger.add("diadem_log.log", diagnose=False)

    return logger


@click.group("main_cli")
def main_cli() -> None:
    """Main CLI entry point for diadem.

    This takes care of parsing the arguments, setting up the logger and runnin
    the DIA mode of DIAdem.
    """
    pass


@main_cli.command(help="Runs the search module of DIAdem")
@click.option(
    "--data_path",
    help="mzML or .d file to use as an input for the search",
)
@click.option("--fasta", help="fasta file to use as an input")
@click.option("--out_prefix", help="Prefix to add to all output files")
@click.option(
    "--mode",
    type=click.Choice(["dda", "dia"], case_sensitive=False),
    default="dia",
)
@click.option("--config", help="Path to the config toml configuration file to use.")
def search(data_path, fasta, out_prefix, mode, config) -> None:
    setup_logger()
    if config:
        config = DiademConfig.from_toml(config)
    else:
        logger.warning("No config path was passed, will use the default")
        config = DiademConfig()
    if mode == "dia":
        diadem_main(
            fasta_path=fasta,
            data_path=data_path,
            config=config,
            out_prefix=out_prefix,
        )
    elif mode == "dda":
        dda_main(
            mzml_path=data_path,
            fasta_path=fasta,
            config=config,
            out_prefix=out_prefix,
        )
    else:
        raise NotImplementedError


@main_cli.command(help="Runs the indexing module of DIAdem")
def index() -> None:
    pass


@main_cli.command(help="Runs the chromatogram alignment module of DIAdem")
def align() -> None:
    pass


@main_cli.command(help="Runs the quantification module of DIAdem")
def quant() -> None:
    pass


if __name__ == "__main__":
    main_cli()
