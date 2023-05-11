import argparse
from dataclasses import replace

from diadem import cli
from diadem.config import DiademConfig

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to the config file")
parser.add_argument("--fasta", type=str, help="Path to the FASTA file")
parser.add_argument("--ms_data", type=str, help="Path to the MS data file")
parser.add_argument("--output", type=str, help="Path to the output file path")
parser.add_argument("--threads", type=int, help="Number of threads to use")


if __name__ == "__main__":
    args, unk = parser.parse_known_args()
    if unk:
        raise ValueError(f"Unrecognized arguments: {unk}")

    config = DiademConfig.from_toml(args.config)
    config = replace(config, run_parallelism=args.threads)

    cli.setup_logger()
    cli.diadem_main(
        fasta_path=args.fasta,
        data_path=args.ms_data,
        config=config,
        out_prefix=args.output,
    )
