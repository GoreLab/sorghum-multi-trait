
#------------------------------------------To process raw data-----------------------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_data_processing.py & 


#------------------------------------To perform first stage analysis-----------------------------------------#

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
Rscript ${PREFIX_code}/mtrait_first_step_analysis.R & 


#--------------------------------To plot results from first stage analysis-----------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_first_step_analysis_plots.py & 


#-----------------------------------Joint races from different databases-------------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_merge_race_information.py & 

#----------------------To split the data into different subsets for cross-validation-------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_cross_validation_data_split.py & 


#------------------To create directories for the ouputs from the cross-validation analysis-------------------#

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Prefix of the output directory is:
PREFIX_out=/workdir/jp2476/repo/resul_mtrait-proj

# Transforming the .sh file into executable:
chmod +755 ${PREFIX_code}/mtrait_cross_validation_output_directories.sh 

# Creating directories:
${PREFIX_code}/mtrait_cross_validation_output_directories.sh -d ${PREFIX_out}


#-------------Creating a text files for mapping the desired set of analysis for cv1 and cv2 schemes----------#

# Setting directory where the data is:
cd /workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation

# Listing training data files names related to the CV1 scheme and storing it for latter usage:
ls y*cv1*trn* > y_cv1_bn_trn_files.txt
ls x*cv1*trn* > x_cv1_bn_trn_files.txt
ls y*cv1*height*trn* > y_cv1_dbn_trn_files.txt
ls x*cv1*height*trn* > x_cv1_dbn_trn_files.txt

# Listing training files names related to the CV1 scheme and storing it for latter usage on PBN model
grep "drymass" y_cv1_bn_trn_files.txt > tmp1.txt 
grep "height" y_cv1_bn_trn_files.txt > tmp2.txt 
paste -d'&' tmp1.txt tmp2.txt > y_cv1_pbn_trn_files.txt
grep "drymass" x_cv1_bn_trn_files.txt > tmp1.txt 
grep "height" x_cv1_bn_trn_files.txt > tmp2.txt 
paste -d'&' tmp1.txt tmp2.txt > x_cv1_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# Listing training phenotypic data files names related to the CV2 scheme and storing it for latter usage (BN):
ls y*cv2*only*trn* > y_cv2_bn_trn_files.txt
ls x*cv2*only*trn* > x_cv2_bn_trn_files.txt

# Listing training phenotypic data files names related to the CV2 scheme and storing it for latter usage (PBN):
echo 'y_cv2_drymass_trn.csv' > tmp1.txt
echo 'x_cv2_drymass_trn.csv' > tmp2.txt
for i in $(seq 1 5); do 

	echo 'y_cv2_drymass_trn.csv' >> tmp1.txt
	echo 'x_cv2_drymass_trn.csv' >> tmp2.txt

done;
paste -d'&' tmp1.txt y_cv2_bn_trn_files.txt > y_cv2_pbn_trn_files.txt
paste -d'&' tmp2.txt x_cv2_bn_trn_files.txt > x_cv2_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# Listing training data files names related to the CV2 scheme and storing it for latter usage (DBN):
ls y*cv2*height*trn* > y_cv2_dbn_trn_files.txt
ls x*cv2*height*trn* > x_cv2_dbn_trn_files.txt
sed -i '/only/d' y_cv2_dbn_trn_files.txt
sed -i '/only/d' x_cv2_dbn_trn_files.txt

# Creating a text file to store the different types of Dynamic Bayesian network models for latter usage (DBN);
echo "DBN-0~5" > dbn_models_cv2_list.txt
echo "DBN-0~1" >> dbn_models_cv2_list.txt
echo "DBN-0~2" >> dbn_models_cv2_list.txt
echo "DBN-0~3" >> dbn_models_cv2_list.txt
echo "DBN-0~4" >> dbn_models_cv2_list.txt

# Listing training files names related to the CV3 scheme and storing it for latter usage on PBN model
for i in 30 45 60 75 90 105
do
	grep "drymass" y_cv1_bn_trn_files.txt > y_tmp1.txt 
	grep "drymass" x_cv1_bn_trn_files.txt > x_tmp1.txt 
    for j in $(seq 1 5)
    do 
      	echo "y_cv2-${i}~only_height_trn.csv" >> y_tmp2.txt
      	echo "x_cv2-${i}~only_height_trn.csv" >> x_tmp2.txt
    done
    paste -d'&' y_tmp1.txt y_tmp2.txt >> y_cv3_pbn_trn_files.txt
    paste -d'&' x_tmp1.txt x_tmp2.txt >> x_cv3_pbn_trn_files.txt
	rm y_tmp1.txt y_tmp2.txt x_tmp1.txt x_tmp2.txt 
done


#-----------------To perform cross-validation analysis using the Bayesian Network model (CV1)----------------#

# Number of analysis:
n_analysis=10

for i in $(seq 1 ${n_analysis}); do  

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv1_bn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv1_bn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='BN'

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Defining the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-----------------To perform cross-validation analysis using the Bayesian Network model (CV2)----------------#

# Number of analysis:
n_analysis=6

for i in $(seq 1 ${n_analysis}); do  

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv2_bn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv2_bn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='BN'

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Defining the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-----------To perform cross-validation analysis using the Pleiotropic Bayesian Network model (CV1)----------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv1_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv1_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='PBN'

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Getting the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'_' -f2 <<<"$y")"
	tmp2="$(cut -d'-' -f1 <<<"$y")"
	tmp2="$(cut -d'_' -f3 <<<"$tmp2")"
	tmp3="$(cut -d'&' -f2 <<<"$y")"
	tmp3="$(cut -d'_' -f3 <<<"$tmp3")"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Creating the name of the output directory:
	dir_out=${PREFIX}/${tmp1}/${tmp2}-${tmp3}/${cv}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-----------To perform cross-validation analysis using the Pleiotropic Bayesian Network model (CV2)----------#

# Number of analysis:
n_analysis=6

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv2_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv2_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='PBN'

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Getting the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'_' -f5 <<<"$y")"
	tmp2="$(cut -d'-' -f1 <<<"$y")"
	tmp2="$(cut -d'_' -f3 <<<"$tmp2")"
	tmp3="$(cut -d'_' -f6 <<<"$y")"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Creating the name of the output directory:
	dir_out=${PREFIX}/${tmp1}/${tmp2}-${tmp3}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;

#-----------To perform cross-validation analysis using the Pleiotropic Bayesian Network model (CV3)----------#

## First batch run (uses 40 cores):
for i in $(seq 1 10); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv3_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv3_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='PBN'

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Getting the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'-' -f2 <<<"$y")"
	tmp1="$(cut -d'~' -f1 <<<"$tmp1")"
	tmp2="$(cut -d'_' -f3 <<<"$y")"
	tmp3="$(cut -d'_' -f7 <<<"$y")"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Creating the name of the output directory:
	dir_out=${PREFIX}/"cv3-"${tmp1}"~only"/${tmp2}-${tmp3}/${cv}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;

## Second batch run (uses 40 cores):
for i in $(seq 11 20); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv3_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv3_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='PBN'

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Getting the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'-' -f2 <<<"$y")"
	tmp1="$(cut -d'~' -f1 <<<"$tmp1")"
	tmp2="$(cut -d'_' -f3 <<<"$y")"
	tmp3="$(cut -d'_' -f7 <<<"$y")"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Creating the name of the output directory:
	dir_out=${PREFIX}/"cv3-"${tmp1}"~only"/${tmp2}-${tmp3}/${cv}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;

## Third batch run (uses 40 cores):
for i in $(seq 21 30); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv3_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv3_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='PBN'

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

	# Getting the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'-' -f2 <<<"$y")"
	tmp1="$(cut -d'~' -f1 <<<"$tmp1")"
	tmp2="$(cut -d'_' -f3 <<<"$y")"
	tmp3="$(cut -d'_' -f7 <<<"$y")"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Creating the name of the output directory:
	dir_out=${PREFIX}/"cv3-"${tmp1}"~only"/${tmp2}-${tmp3}/${cv}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;



#-------------To perform cross-validation (CV1) analysis using the Dynamic Bayesian Network model------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv1_dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv1_dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='DBN-0~6'

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Defining the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-------------To perform cross-validation (CV2) analysis using the Dynamic Bayesian Network model------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv2_dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv2_dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model=$(sed -n "${i}p" ${dir_in}/dbn_models_cv2_list.txt)

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

	# Defining the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#----------------------------------------Install python locally----------------------------------------------#

# Download anaconda:
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh

# For install python local:
bash Anaconda3-5.1.0-Linux-x86_64.sh

# Folder where the python is installed:
/workdir/jp2476/software/python

# To install modules:
/workdir/jp2476/software/python/bin/pip install SOFTWARE_NAME --upgrade

# To fire bash:
/workdir/jp2476/software/python/bin/python/bash_2layers.py > out.txt

## Bug-solve instruction in tensorflow:
# when saving the output from the tensorflow,
# define the dir first, then do ./ on the dir arg of tf_saver


#---------------------------------For killing all process from the user--------------------------------------#

# For kill all process of a user:
screen -X -S dnn quit

pkill -u jp2476


#------------------------------Temporary bash code written in the project------------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing a flag test:
${PREFIX_python}/python ${PREFIX_code}/tmp.py --model bn


############################# Working on MLM model CV1 ###################################

i=1

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Name of the file with the phenotypes:
y=$(sed -n "${i}p" ${dir_in}/y_cv1_bn_trn_files.txt)

# Name of the file with the features:
x=$(sed -n "${i}p" ${dir_in}/x_cv1_bn_trn_files.txt)

# Name of the model: Multivariate Linear Mixed (MLM) model
model='MTiLM'

# Getting the current fold of the cross-validation:
cv="$(cut -d'_' -f4 <<<"$y")"

# Directory of the project folder:
dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

# Prefix of the output directory:
PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"

# Define the output directory for the outputs:
dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

# Run the code:
Rscript ./mtrait_mixed_models.R ${y} ${x} ${m} ${dir_in} ${dir_proj} ${dir_out};







i=1

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Name of the file with the phenotypes:
y=$(sed -n "${i}p" ${dir_in}/y_cv1_dbn_trn_files.txt)

# Name of the file with the features:
x=$(sed -n "${i}p" ${dir_in}/x_cv1_dbn_trn_files.txt)

# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
model='MTiLM-0~6'

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Directory of the project folder:
dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

# Getting just the model prefix:
tmp="$(cut -d'-' -f1 <<<"$model")"

# Prefix of the output directory:
PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

# Getting the current fold of the cross-validation:
cv="$(cut -d'_' -f4 <<<"$y")"

# Defining the output directory for the outputs:
dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

# Run the code:
Rscript ./mtrait_mixed_models.R ${y} ${x} ${m} ${dir_in} ${dir_proj} ${dir_out};

# Sleep for avoid exploding several processes:
sleep 5


