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
mkdir ${PREFIX}/outputs/cross_validation/BN/cv1/drymass;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~45;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~45/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~45/drymass;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~60;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~60/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~60/drymass;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~75;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~75/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~75/drymass;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~90;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~90/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~90/drymass;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~105;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~105/height;
mkdir ${PREFIX}/outputs/cross_validation/BN/cv2-30~105/drymass;

# Creating CV directories on Pleiotropic Bayesian network output folder:
mkdir ${PREFIX}/outputs/cross_validation/PBN;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv1/drymass;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~45;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~45/height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~45/drymass;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~60;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~60/height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~60/drymass;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~75;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~75/height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~75/drymass;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~90;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~90/height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~90/drymass;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~105;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~105/height;
mkdir ${PREFIX}/outputs/cross_validation/PBN/cv2-30~105/drymass;

# Creating CV directories on Dynamic Bayesian network output folder:
mkdir ${PREFIX}/outputs/cross_validation/DBN;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv1/drymass;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~45;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~45/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~45/drymass;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~60;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~60/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~60/drymass;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~75;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~75/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~75/drymass;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~90;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~90/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~90/drymass;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~105;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~105/height;
mkdir ${PREFIX}/outputs/cross_validation/DBN/cv2-30~105/drymass;
