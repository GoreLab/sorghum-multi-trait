
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
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV1" -bt "starch" &
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV1" -bt "drymass" &
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV2.30~45" & 
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV2.30~60" & 
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV2.30~75" & 
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV2.30~90" & 
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_data.py -cv "CV2.30~105" & 


#---------------------------- To run bayesian models--------------------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_bayesian_models.py -c 1 -d "cv1_height" -m "BN" -cv "CV1" & 

${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_bayesian_models.py -c 2 -d "cv1_biomass_drymass" -m "BN" -cv "CV1" & 

${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_bayesian_models.py -c 2 -d "cv1_biomass_starch" -m "BN" -cv "CV1" & 

${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_bayesian_models.py -c 1 -d "cv1_biomass-cv1_height" -m "PBN0" -cv "CV1" & 

${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_bayesian_models.py -c 1 -d "cv1_biomass-cv1_height" -m "PBN1" -cv "CV1" & 


#-------------------------- For running dnn script -----------------------------------#

## Biomass analysis:

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Number of processors:
n_proc=40

# Looping over codes:
for i in $(seq 0 $((n_proc-1))); do  
	
	# Firing process:
	${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_dnn_models.py -c ${i} -nalt 10 -d "cv1_biomass_drymass" -m "DNN" -cv "CV1" &

	# Wait some time to fire the next process:
	sleep 10
done;

# Looping over codes:
for i in $(seq 0 $((n_proc-1))); do  
	
	# Firing process:
	${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_dnn_models.py -c ${i} -nalt 10 -d "cv1_biomass_starch" -m "DNN" -cv "CV1" &

	# Wait some time to fire the next process:
	sleep 10
done;

# Looping over codes:
for i in $(seq 0 $((n_proc-1))); do  
	
	# Firing process:
	${PREFIX_python}/python ${PREFIX_code}/mtrait_cv_dnn_models.py -c ${i} -nalt 10 -d "cv1_height" -m "DNN" -cv "CV1" &

	# Wait some time to fire the next process:
	sleep 10
done;


#---------------------------- To run rrblup models--------------------------------------#

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
Rscript ${PREFIX_code}/mtrait_cv_rrblup_model.R & 

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
screen -X -S dnn quit

pkill -u jp2476


#--------------- Temporary bash code written in the project --------------------------#

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing a flag test:
${PREFIX_python}/python ${PREFIX_code}/tmp.py --model bn



