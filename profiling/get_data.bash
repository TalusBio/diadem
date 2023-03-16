#!/bin/bash

mkdir -p profiling_data
aws s3 cp --profile mfa s3://data-pipeline-mzml-bucket/221229_ChessFest_Plate3/Chessfest_Plate3_RH4_DMSO_DIA.mzML.gz ./profiling_data/.
aws s3 cp --profile mfa s3://data-pipeline-metadata-bucket/uniprot_human_sp_canonical_2021-11-19_crap.fasta ./profiling_data/.
curl ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD028735/LFQ_timsTOFPro_PASEF_Ecoli_01.d.zip --output ./profiling_data/ecoli_timsTOFPro_PASEF.d.zip
curl ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD028735/LFQ_timsTOFPro_diaPASEF_Ecoli_01.d.zip --output ./profiling_data/ecoli_timsTOFPro_diaPASEF.d.zip
curl https://ftp.pride.ebi.ac.uk/pride/data/archive/2021/07/PXD027359/20210510_TIMS03_EVO03_PaSk_SA_HeLa_50ng_5_6min_DIA_high_speed_S1-B2_1_25186.d.zip --output ./profiling_data/hela_5min_diaPASEF.d.zip
# gunzip ./profiling_data/*.gz
for i in  ./profiling_data/*.zip ; do unzip $i ; done
mv LFQ_timsTOFPro_* ./profiling_data/.
mv 20210510_TIMS03_EVO03* ./profiling_data/.

alphatims export hdf profiling_data/20210510_TIMS03_EVO03_PaSk_SA_HeLa_50ng_5_6min_DIA_high_speed_S1-B2_1_25186.d

curl https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/UP000000625/UP000000625_83333.fasta.gz --output ./profiling_data/UP000000625_83333.fasta.gz
gunzip ./profiling_data/UP000000625_83333.fasta.gz
curl https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz --output ./profiling_data/UP000005640_9606.fasta.gz
gunzip ./profiling_data/UP000005640_9606.fasta.gz
