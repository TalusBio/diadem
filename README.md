
![GitHub](https://img.shields.io/github/license/TalusBio/diadem?style=flat-square)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/talusbio/diadem?style=flat-square)


# diadem

A feature-centric DIA search engine

## Current stage

Under development. We are happy yo take ideas and questions! Feel free to start an issue!

## Installation

```shell
git clone git@github.com:TalusBio/diadem.git
cd diadem
pip install .
# pip install -e ".[dev,profiling,test]" # for development install
```

## Usage

### Command line usage

```shell
# All commands
diadem --help

# See the help for the search functionality
diadem search --help
diadem --data_path {myfile.mzML/myfile.d/myfile.hdf} \
    --fasta myfasta.fasta \
    --out_prefix my_directory/my_results \
    --mode dia \
    --config myconfig.toml
```

## Release milestones

- [ ] Quantification module
- [ ] Stable quant module
- [ ] RT alignment module
