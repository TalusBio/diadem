#!/bin/bash

mkdir -p profiling_data

## Fasta Files
# Ecoli
curl https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/UP000000625/UP000000625_83333.fasta.gz --output ./profiling_data/UP000000625_83333.fasta.gz
gunzip ./profiling_data/UP000000625_83333.fasta.gz

# Human
curl https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz --output ./profiling_data/UP000005640_9606.fasta.gz
gunzip ./profiling_data/UP000005640_9606.fasta.gz
aws s3 cp --profile mfa s3://data-pipeline-metadata-bucket/contaminants.fasta ./profiling_data/.

cat ./profiling_data/contaminants.fasta ./profiling_data/UP000000625_83333.fasta >> ./profiling_data/UP000000625_83333_crap.fasta
cat ./profiling_data/contaminants.fasta ./profiling_data/UP000005640_9606.fasta >> ./profiling_data/UP000005640_9606_crap.fasta

# ./profiling_data/UP000000625_83333_crap.fasta
# ./profiling_data/UP000005640_9606_crap.fasta

# Raw Data
# Ecoli
# TimsTof
curl ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD028735/LFQ_timsTOFPro_diaPASEF_Ecoli_01.d.zip --output ./profiling_data/ecoli_timsTOFPro_diaPASEF.d.zip

# Human
# Orbi
aws s3 cp --profile mfa s3://tmp-jspp-diadem-assets/220119_hela_44m_1.mzML.gz

# TimsTof


for i in  ./profiling_data/*.zip ; do unzip $i -d profiling_data ; done

# This is done in docker ... still waiting for the mann lab to check my PR
docker build -t alphatims_docker .
docker run -rm -it -v ${PWD}/profiling_data/:/data/ alphatims export hdf LFQ_timsTOFPro_diaPASEF_Ecoli_01
