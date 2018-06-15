
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

# Listing training phenotypic data files names related to the CV1 scheme and storing it for latter usage:
ls y*cv1*trn* > y_cv1_bn-dbn_trn_files.txt

# Listing training genotypic data files names related to the CV1 scheme and storing it for latter usage:
ls x*cv1*trn* > x_cv1_bn-dbn_trn_files.txt

# Listing training files names related to the CV1 scheme and storing it for latter usage on PBN model
grep "drymass" y_cv1_bn-dbn_trn_files.txt > tmp1.txt 
grep "height" y_cv1_bn-dbn_trn_files.txt > tmp2.txt 
paste -d'-' tmp1.txt tmp2.txt > y_cv1_pbn_trn_files.txt
grep "drymass" x_cv1_bn-dbn_trn_files.txt > tmp1.txt 
grep "height" x_cv1_bn-dbn_trn_files.txt > tmp2.txt 
paste -d'-' tmp1.txt tmp2.txt > x_cv1_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# Listing training phenotypic data files names related to the CV2 scheme and storing it for latter usage:
ls y*cv2*trn* > y_cv2_dbn_trn_files.txt

# Listing training genotypic data files names related to the CV2 scheme and storing it for latter usage:
ls x*cv2*trn* > x_cv2_dbn_trn_files.txt

# Creating a text file to store the different types of Dynamic Bayesian network models for latter usage;
echo 'DBN-0~1' > dbn_models_cv2_list.txt
echo "DBN-0~2" >> dbn_models_cv2_list.txt
echo "DBN-0~3" >> dbn_models_cv2_list.txt
echo "DBN-0~4" >> dbn_models_cv2_list.txt
echo "DBN-0~5" >> dbn_models_cv2_list.txt

#########* Temp code to use below latter:

# Number of analysis to fire:
n_analysis=10

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

for i in $(seq 1 ${n_analysis}); do  

	tmp1=$(sed -n "${i}p" ${dir_in}/y_cv1_bn-dbn_trn_files.txt)
	tmp2=$(sed -n "${i}p" ${dir_in}/x_cv1_bn-dbn_trn_files.txt)

	echo $tmp1
	echo $tmp2
	echo "-----"

done;


#-------------------To perform cross-validation analysis using the Bayesian Network model--------------------#

# Number of analysis:
n_analysis=10

for i in $(seq 1 ${n_analysis}); do  

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv1_bn-dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv1_bn-dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='BN'

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

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


#--------------To perform cross-validation analysis using the Pleiotropic Bayesian Network model-------------#

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

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${model}"
		
	# Getting the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'_' -f2 <<<"$y")"
	tmp2="$(cut -d'-' -f1 <<<"$y")"
	tmp2="$(cut -d'_' -f3 <<<"$tmp2")"
	tmp3="$(cut -d'-' -f2 <<<"$y")"
	tmp3="$(cut -d'_' -f3 <<<"$tmp3")"

	# Creating the name of the output directory:
	dir_out=${PREFIX}/${tmp1}/${tmp2}-${tmp3}

	# Prefix for running the script:
	PREFIX_python=/workdir/jp2476/software/python/bin

	# Running the code:
	${PREFIX_python}/python ${dir_proj}/codes/mtrait_bayesian_networks.py -y ${y} -x ${x} -m ${model} -di ${dir_in} -dp ${dir_proj} -do ${dir_out} & 

done;

#----------------To perform cross-validation analysis using the Dynamic Bayesian Network model---------------#

# Name of the file with the phenotypes:
y="y_cv2-30~45_height_trn.csv"

# Name of the file with the features:
x="x_cv2-30~45_height_trn.csv"

# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
model='DBN-0~1'

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



s=y_cv1_drymass_k1_trn.csv

echo "$(cut -d'_' -f3 <<<"$s")"
