
#--------------------------- To process raw data--------------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_data_processing.py & 

#---------------------------- To process cv data--------------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py & 


#---------------------------- To run bayesian models--------------------------------------#

# Parameters (height run):
core=1 
data="cv1_height"
model="BN"
cv="CV1"

# Parameters (biomass run):
core=2 
data="cv1_biomass"
model="BN"
cv="CV1"

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_bayesian_models.py -c ${core} -d ${data} -m ${model} -cv ${cv} & 


#-------------------------- For running dnn script -----------------------------------#

# Prefix python:
# PREFIX_python=/home/aafgarci/anaconda3/bin
# PREFIX_python=/workdir/jp2476/software/python/bin
PREFIX_python=/home/jhonathan/Documents/python/bin

# Prefix code:
# PREFIX_code=/data1/aafgarci/jhonathan/sorghum-multi-trait/codes
# PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes
PREFIX_code=/home/jhonathan/Documents/sorghum-multi-trait/codes

# Number of processes:
n_proc=7

# Number of alternatives:
n_alt=143

# Looping over codes:
for i in $(seq 0 $((n_proc-1))); do  
	
	# Firing process:
	${PREFIX_python}/python ${PREFIX_code}/mtrait_dnn_models.py -c ${i} -nalt ${n_alt} &

	# Wait some time to fire the next process:
	sleep 1

done;


#-------------------------- Install python locally -----------------------------------#

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


#------------------ For killing all process from the user ----------------------------#

# For kill all process of a user:
screen -X -S jhonathan quit

pkill -u aafgarci


#--------------- Temporary bash code written in the project --------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing a flag test:
${PREFIX_python}/python ${PREFIX_code}/tmp.py --model bn



