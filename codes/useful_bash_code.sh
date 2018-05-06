
########################## For running python scripts ##################################

## For firing python scripts into bash code:

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_data_processing.py & 


############################ For running dbn script ###################################

## For firing python scripts into bash code:

# Prefix python:
PREFIX_python=/workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code=/workdir/jp2476/repo/sorghum-multi-trait/codes

# Number of processes:
n_proc=15
n_alt=70

# Looping over codes:
for i in $(seq 0 $((n_proc-1))); do  
	
	# Firing process:
	${PREFIX_python}/python ${PREFIX_code}/mtrait_cross_validation_and_models.py -c ${i} -nalt ${n_alt} &

	# Wait some time to fire the next process:
	sleep 1

done;





############################ Install python locally ####################################

# For install python local:
bash ./Anaconda3-5.0.1-Linux-x86_64.sh

# Folder where the python is installed:
/workdir/jp2476/software/python

# To install modules:
/workdir/jp2476/software/python/bin/pip install SOFTWARE_NAME --upgrade

# To fire bash:
/workdir/jp2476/software/python/bin/python/bash_2layers.py > out.txt

# Bug-solve instruction in tensorflow:
# when saving the output from the tensorflow,
# define the dir first, then do ./ on the dir arg of tf_saver


#########################################################################################



