
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


#-------------------To perform cross-validation analysis using the Bayesian Network model--------------------#

# Name of the file with the phenotypes:
y="y_cv1_height_k0_trn.csv"

# Name of the file with the features:
x="x_cv1_height_k0_trn.csv"

# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
model='BN'

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

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


#--------------To perform cross-validation analysis using the Pleiotropic Bayesian Network model-------------#

# Name of the file with the phenotypes:
y="y_cv1_drymass_k0_trn.csv-y_cv1_height_k0_trn.csv"

# Name of the file with the features:
x="x_cv1_drymass_k0_trn.csv-x_cv1_height_k0_trn.csv"

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


#----------------To perform cross-validation analysis using the Dynamic Bayesian Network model---------------#

# Name of the file with the phenotypes:
y="y_cv1_height_k0_trn.csv"

# Name of the file with the features:
x="x_cv1_height_k0_trn.csv"

# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
model='DBN-0~5'

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
