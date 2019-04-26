#!/bin/bash

# Reading the flag:
while getopts ":p:" opt; do
  case $opt in
    p)
      ROOT_PATH=$OPTARG;
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Create the folder for receiving the outputs of the codes:
mkdir ${ROOT_PATH}/output_sorghum-multi-trait;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/processed_data;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/heritabilities;

# Create CV directories on Bayesian network (BN) output folder:
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/height/k0;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/height/k1;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/height/k2;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/height/k3;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/height/k4;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/drymass;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/drymass/k0;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/drymass/k1;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/drymass/k2;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/drymass/k3;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/cv5f/drymass/k4;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-30~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-30~only/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-45~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-45~only/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-60~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-60~only/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-75~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-75~only/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-90~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-90~only/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-105~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/BN/fcv-105~only/height;

# Create CV directories on Pleiotropic Bayesian network (PBN) output folder:
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f/drymass-height/k0;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f/drymass-height/k1;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f/drymass-height/k2;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f/drymass-height/k3;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/cv5f/drymass-height/k4;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-30~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-30~only/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-45~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-45~only/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-60~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-60~only/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-75~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-75~only/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-90~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-90~only/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-105~only;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/PBN/fcv-105~only/drymass-height;

# Create CV directories on Dynamic Bayesian network (DBN) output folder:
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f/height/k0;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f/height/k1;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f/height/k2;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f/height/k3;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/cv5f/height/k4;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~45;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~45/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~60;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~60/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~75;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~75/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~90;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~90/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~105;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/DBN/fcv-30~105/height;

# Create CV directories for the Multiple Time Linear Mixed model (MTi-GBLUP) output folder:
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f/height/k0;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f/height/k1;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f/height/k2;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f/height/k3;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/cv5f/height/k4;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~45;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~45/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~60;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~60/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~75;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~75/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~90;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~90/height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~105;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTi-GBLUP/fcv-30~105/height;

# Create CV directories for the Multiple Trait Linear Mixed model (MTr-GBLUP) output folder:
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f/drymass-height/k0;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f/drymass-height/k1;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f/drymass-height/k2;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f/drymass-height/k3;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/cv5f/drymass-height/k4;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~45;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~45/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~60;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~60/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~75;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~75/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~90;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~90/drymass-height;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~105;
mkdir ${ROOT_PATH}/output_sorghum-multi-trait/cv/MTr-GBLUP/fcv-30~105/drymass-height;

