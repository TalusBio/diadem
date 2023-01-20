import argparse
import sys

from loguru import logger

from diadem.config import DiademConfig
from diadem.search.diadem import diadem_main

parser = argparse.ArgumentParser(
    "diadem",
    description="diadem prototype",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers()

parser_append = subparsers.add_parser("config")
parser_append = subparsers.add_parser("index")
parser_append = subparsers.add_parser("search")

# diadem search --mode
parser_append = subparsers.add_parser("")

parser.add_argument("--mzml_file", help="mzML file to use as an input for the search")
parser.add_argument("--fasta", help="fasta file to use as an input")
parser.add_argument("--out_prefix", help="Prefix to add to all output files")


def main_cli(*args: str) -> None:
    """Main CLI entry point for diadem.

    This takes care of parsing the arguments, setting up the logger and runnin
    the DIA mode of DIAdem.

    Parameters
    ----------
    args : str
        Arguments to pass to the parser in addition to the
        command line arguments.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.add("diadem_log.log", diagnose=False)
    args, unkargs = parser.parse_known_args(*args)
    if unkargs:
        logger.error(f"Unknown args: {unkargs}")
        raise ValueError(f"Unknown args: {unkargs}")

    # TODO add a config file parser addition ...
    config = DiademConfig()
    diadem_main(
        fasta_path=args.fasta,
        mzml_path=args.mzml_file,
        config=config,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main_cli()
