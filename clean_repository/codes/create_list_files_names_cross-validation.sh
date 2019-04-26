#!/bin/bash

# Reading the flag:
while getopts ":o:" opt; do
  case $opt in
    o)
      OUT_PATH=$OPTARG;
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

# Set directory where the data is:
cd ${OUT_PATH}/processed_data

# List training data files names related to the cv5f scheme and storing it for latter usage:
ls y*cv5f*trn* > y_cv5f_bn_trn_files.txt
ls x*cv5f*trn* > x_cv5f_bn_trn_files.txt
ls y*cv5f*height*trn* > y_cv5f_dbn_trn_files.txt
ls x*cv5f*height*trn* > x_cv5f_dbn_trn_files.txt

# List training files names related to the cv5f scheme and storing it for latter usage on PBN model
grep "drymass" y_cv5f_bn_trn_files.txt > tmp1.txt 
grep "height" y_cv5f_bn_trn_files.txt > tmp2.txt 
paste -d'&' tmp1.txt tmp2.txt > y_cv5f_pbn_trn_files.txt
grep "drymass" x_cv5f_bn_trn_files.txt > tmp1.txt 
grep "height" x_cv5f_bn_trn_files.txt > tmp2.txt 
paste -d'&' tmp1.txt tmp2.txt > x_cv5f_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# List training phenotypic data files names related to the fcv scheme and storing it for latter usage (BN):
ls y*fcv*only*trn* > y_fcv_bn_trn_files.txt
ls x*fcv*only*trn* > x_fcv_bn_trn_files.txt

# List training phenotypic data files names related to the fcv scheme and storing it for latter usage (PBN):
echo 'y_fcv_drymass_trn.csv' > tmp1.txt
echo 'x_fcv_drymass_trn.csv' > tmp2.txt
for i in $(seq 1 5); do 

  echo 'y_fcv_drymass_trn.csv' >> tmp1.txt
  echo 'x_fcv_drymass_trn.csv' >> tmp2.txt

done;
paste -d'&' tmp1.txt y_fcv_bn_trn_files.txt > y_fcv_pbn_trn_files.txt
paste -d'&' tmp2.txt x_fcv_bn_trn_files.txt > x_fcv_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# List training data files names related to the fcv scheme and storing it for latter usage (DBN):
ls y*fcv*height*trn* > y_fcv_dbn_trn_files.txt
ls x*fcv*height*trn* > x_fcv_dbn_trn_files.txt
sed -i '/only/d' y_fcv_dbn_trn_files.txt
sed -i '/only/d' x_fcv_dbn_trn_files.txt

# Create a text file to store the different types of Dynamic Bayesian network models for latter usage (DBN);
echo "DBN-0~5" > dbn_models_fcv_list.txt
echo "DBN-0~1" >> dbn_models_fcv_list.txt
echo "DBN-0~2" >> dbn_models_fcv_list.txt
echo "DBN-0~3" >> dbn_models_fcv_list.txt
echo "DBN-0~4" >> dbn_models_fcv_list.txt

# List of files to get train data index into R to perform Multivariate GBLUP models analysis in R:
cat y_cv5f_dbn_trn_files.txt > cv5f_mti-gblup_files.txt
cat y_cv5f_pbn_trn_files.txt > cv5f_mtr-gblup_files.txt
cat y_fcv_dbn_trn_files.txt > fcv_mti-gblup_files.txt
echo 'y_fcv_drymass_trn.csv' > tmp1.txt
for i in $(seq 1 4); do 

  echo 'y_fcv_drymass_trn.csv' >> tmp1.txt

done;
paste -d'&' tmp1.txt y_fcv_dbn_trn_files.txt > fcv_mtr-gblup_files.txt
rm tmp1.txt

# Create a text file to store the different types of Multi Time GBLUP(MTiGBLUP) models for latter usage;
echo "MTiGBLUP-0~5" > mti-gblup_models_fcv_list.txt
echo "MTiGBLUP-0~1" >> mti-gblup_models_fcv_list.txt
echo "MTiGBLUP-0~2" >> mti-gblup_models_fcv_list.txt
echo "MTiGBLUP-0~3" >> mti-gblup_models_fcv_list.txt
echo "MTiGBLUP-0~4" >> mti-gblup_models_fcv_list.txt

# Create a text file to store the different types of Multi Trait GBLUP (MTrGBLUP) models for latter usage;
echo "MTrGBLUP-0~5" > mtr-gblup_models_fcv_list.txt
echo "MTrGBLUP-0~1" >> mtr-gblup_models_fcv_list.txt
echo "MTrGBLUP-0~2" >> mtr-gblup_models_fcv_list.txt
echo "MTrGBLUP-0~3" >> mtr-gblup_models_fcv_list.txt
echo "MTrGBLUP-0~4" >> mtr-gblup_models_fcv_list.txt

