#!/bin/bash

# Reading the flag:
while getopts ":d:" opt; do
  case $opt in
    d)
      echo "Creating directories for receiving the Bayesian Network outputs" >&2;
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

# Creating CV directories on Bayesian network output folder:
mkdir ${PREFIX}/outputs/cross_validation/BN;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/height/k0;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/height/k1;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/height/k2;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/height/k3;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/height/k4;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass/k0;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass/k1;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass/k2;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass/k3;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass/k4;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~only;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~only/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-45~only;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-45~only/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-60~only;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-60~only/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-75~only;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-75~only/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-90~only;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-90~only/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-105~only;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-105~only/height;

# Creating CV directories on Pleiotropic Bayesian network output folder:
mkdir ${PREFIX}/outputs/cross_validation/PBN;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass-height/k0;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass-height/k1;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass-height/k2;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass-height/k3;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass-height/k4;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~only;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~only/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-45~only;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-45~only/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-60~only;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-60~only/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-75~only;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-75~only/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-90~only;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-90~only/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-105~only;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-105~only/drymass-height;

# Creating CV directories on Dynamic Bayesian network output folder:
mkdir ${PREFIX}/outputs/cross_validation/DBN;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height/k0;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height/k1;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height/k2;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height/k3;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height/k4;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~45;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~45/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~60;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~60/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~75;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~75/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~90;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~90/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~105;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~105/height;

# Creating CV directories for the Multiple Time Linear Mixed model output folder:
mkdir ${PREFIX}/outputs/cross_validation/MTiLM;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1/height;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1/height/k0;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1/height/k1;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1/height/k2;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1/height/k3;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv1/height/k4;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~45;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~45/height;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~60;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~60/height;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~75;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~75/height;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~90;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~90/height;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~105;
mkdir ${PREFIX}/outputs/cross_validation/MTiLM/cv2-30~105/height;

# Creating CV directories for the Multiple Trait Linear Mixed model output folder:
mkdir ${PREFIX}/outputs/cross_validation/MTrLM;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1/drymass-height/k0;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1/drymass-height/k1;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1/drymass-height/k2;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1/drymass-height/k3;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv1/drymass-height/k4;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~45;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~45/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~60;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~60/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~75;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~75/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~90;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~90/drymass-height;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~105;
mkdir ${PREFIX}/outputs/cross_validation/MTrLM/cv2-30~105/drymass-height;




