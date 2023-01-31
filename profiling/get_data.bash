#!/bin/bash

mkdir -p profiling_data
aws s3 cp --profile mfa s3://data-pipeline-mzml-bucket/221229_ChessFest_Plate3/Chessfest_Plate3_RH4_DMSO_DIA.mzML.gz ./profiling_data/.
aws s3 cp --profile mfa s3://data-pipeline-metadata-bucket/uniprot_human_sp_canonical_2021-11-19_crap.fasta ./profiling_data/.
gunzip *.gz
