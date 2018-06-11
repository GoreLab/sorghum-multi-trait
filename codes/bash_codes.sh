
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

#--------------------------To create the output from the cross-validation analysis---------------------------#

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Prefix of the output directory is:
PREFIX_out=/workdir/jp2476/repo/resul_mtrait-proj

# Transforming the .sh file into executable:
chmod +755 ${PREFIX_code}/mtrait_cross_validation_output_directories.sh 

# Creating directories:
${PREFIX_code}/mtrait_cross_validation_output_directories.sh -d ${PREFIX_out}


#----------------------------------------Install python locally----------------------------------------------#

# Download anaconta:
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
