import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to the config file")
parser.add_argument("--fasta", type=str, help="Path to the FASTA file")
parser.add_argument("--ms_data", type=str, help="Path to the MS data file")
parser.add_argument("--output", type=str, help="Path to the output file path")
parser.add_argument("--threads", type=int, help="Number of threads to use")
