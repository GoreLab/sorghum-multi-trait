#!/bin/bash

# Reading the flag:
while getopts ":d:" opt; do
  case $opt in
    d)
      echo "Creating directories" >&2;
      PREFIX=$OPTARG;
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

# Create CV directories on Bayesian network (BN) output folder:
mkdir ${PREFIX}/outputs/cv/BN;
mkdir ${PREFIX}/outputs/cv/BN/5fcv;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/height;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/height/k0;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/height/k1;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/height/k2;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/height/k3;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/height/k4;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/drymass;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/drymass/k0;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/drymass/k1;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/drymass/k2;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/drymass/k3;
mkdir ${PREFIX}/outputs/cv/BN/5fcv/drymass/k4;
mkdir ${PREFIX}/outputs/cv/BN/fcv-30~only;
mkdir ${PREFIX}/outputs/cv/BN/fcv-30~only/height;
mkdir ${PREFIX}/outputs/cv/BN/fcv-45~only;
mkdir ${PREFIX}/outputs/cv/BN/fcv-45~only/height;
mkdir ${PREFIX}/outputs/cv/BN/fcv-60~only;
mkdir ${PREFIX}/outputs/cv/BN/fcv-60~only/height;
mkdir ${PREFIX}/outputs/cv/BN/fcv-75~only;
mkdir ${PREFIX}/outputs/cv/BN/fcv-75~only/height;
mkdir ${PREFIX}/outputs/cv/BN/fcv-90~only;
mkdir ${PREFIX}/outputs/cv/BN/fcv-90~only/height;
mkdir ${PREFIX}/outputs/cv/BN/fcv-105~only;
mkdir ${PREFIX}/outputs/cv/BN/fcv-105~only/height;

# Create CV directories on Pleiotropic Bayesian network (PBN) output folder:
mkdir ${PREFIX}/outputs/cv/PBN;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv/drymass-height;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv/drymass-height/k0;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv/drymass-height/k1;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv/drymass-height/k2;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv/drymass-height/k3;
mkdir ${PREFIX}/outputs/cv/PBN/5fcv/drymass-height/k4;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-30~only;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-30~only/drymass-height;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-45~only;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-45~only/drymass-height;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-60~only;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-60~only/drymass-height;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-75~only;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-75~only/drymass-height;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-90~only;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-90~only/drymass-height;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-105~only;
mkdir ${PREFIX}/outputs/cv/PBN/fcv-105~only/drymass-height;

# Create CV directories on Dynamic Bayesian network (DBN) output folder:
mkdir ${PREFIX}/outputs/cv/DBN;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv/height;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv/height/k0;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv/height/k1;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv/height/k2;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv/height/k3;
mkdir ${PREFIX}/outputs/cv/DBN/5fcv/height/k4;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~45;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~45/height;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~60;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~60/height;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~75;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~75/height;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~90;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~90/height;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~105;
mkdir ${PREFIX}/outputs/cv/DBN/fcv-30~105/height;

# Create CV directories for the Multiple Time Linear Mixed model (MTi-GBLUP) output folder:
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv/height;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv/height/k0;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv/height/k1;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv/height/k2;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv/height/k3;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/5fcv/height/k4;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~45;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~45/height;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~60;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~60/height;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~75;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~75/height;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~90;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~90/height;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~105;
mkdir ${PREFIX}/outputs/cv/MTi-GBLUP/fcv-30~105/height;

# Create CV directories for the Multiple Trait Linear Mixed model (MTr-GBLUP) output folder:
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv/drymass-height;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv/drymass-height/k0;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv/drymass-height/k1;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv/drymass-height/k2;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv/drymass-height/k3;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/5fcv/drymass-height/k4;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~45;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~45/drymass-height;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~60;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~60/drymass-height;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~75;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~75/drymass-height;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~90;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~90/drymass-height;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~105;
mkdir ${PREFIX}/outputs/cv/MTr-GBLUP/fcv-30~105/drymass-height;

