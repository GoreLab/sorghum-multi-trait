#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#---------------------------------Tutorial for reproducing the study:----------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#--------Novel Bayesian Networks for Genomic Prediction of Developmental Traits in Biomass Sorghum-----------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#--------------Written by: Jhonathan Pedroso Rigal dos Santos, contact: jhowpd@gmail.com---------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#


#-------------------------------Downloading the data used for the study--------------------------------------#

# Prefix of the output directory is:
ROOT_PATH=/workdir/jp2476

# Download the data used for the study:
Obs--> Link for download: https://de.cyverse.org/de/?type=data&folder=/iplant/home/shared/GoreLab/dataFromPubs/dosSantos_BayesianNetworks_2019

# The data should be save under the following path structure (README.txt contains information of the files):
ROOT_PATH
└── dosSantos_GenomicPrediction_2019
    ├── raw_data
    │   ├── Biomass_2016.csv
    │   ├── Biomass_SF2017.csv
    │   ├── gbs.csv
    │   ├── gbs_info.csv
    │   ├── gbs_unimputed_markers.vcf.gz
    │   ├── genotype_names_corrected.csv
    │   ├── heights_2016.csv
    │   └── heights_SF2017.csv
    └── README.txt

# The repository should also be in root directory:
ROOT_PATH
|── sorghum-multi-trait
└── dosSantos_GenomicPrediction_2019

     
#----------If you do not have sudo permision install python locally (for install packages via pip)-----------#

# Create a path to instal the softwere locally:
mkdir ${ROOT_PATH}/local/

# Install python locally (please provide a path to install python to be informed forward):
cd ${ROOT_PATH}/local/
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh


#-------------------------------------------Install packages-------------------------------------------------#

# Prefix of the output directory is:
ROOT_PATH=/workdir/jp2476

## Set the path where python was locally installed (this is a example of the one informed when installing):
# Obs--> If you have sudo permisions and can install the python packages, just ignore this block
#------> and replace in the bellow codes the ${PYTHON_PATH}/bin/pip or ${PYTHON_PATH}/bin/python
#------> by only pip and python, respectively. You can also provide the path if you wish to
#------> execute pip and python. The Python 3 is required to run the codes. If errors happen 
#------> caused by dependencies we recommend to install python locally following the above instructions.
PYTHON_PATH=${ROOT_PATH}/local/python

# Install python packages (install these versions to avoid errors):
${PYTHON_PATH}/bin/pip install argparse==1.4.0
${PYTHON_PATH}/bin/pip install numpy==1.16.3
${PYTHON_PATH}/bin/pip install pandas==0.24.2
${PYTHON_PATH}/bin/pip install sklearn==0.0
${PYTHON_PATH}/bin/pip install matplotlib==3.0.3
${PYTHON_PATH}/bin/pip install seaborn==0.9.0
${PYTHON_PATH}/bin/pip install pystan==2.17.1.0

# Install R packages (install these versions to avoid errors):
# Obs: asreml v3.0 should already be installed to work the scripts of the phenotypic analysis to obtain adjusted means and heritabilities
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/optparse/optparse_1.6.0.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/data.table/data.table_1.12.0.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/magrittr_1.5.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/stringr/stringr_1.3.1.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/stringr/stringr_1.3.1.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/EMMREML_3.1.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/rrBLUP_4.6.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/tidyr_0.8.3.tar.gz", repos=NULL, type="source")'


#------------------------------------------Create directories------------------------------------------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Set the root directory:
cd ${ROOT_PATH}

# Transform the .sh file into executable:
chmod +755 ${REPO_PATH}/clean_repository/codes/create_directories.sh 

# Create directories:
${REPO_PATH}/clean_repository/codes/create_directories.sh -p ${ROOT_PATH}


#---------------------------------------Phenotypic data analysis---------------------------------------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder (output_sorghum-multi-trait folder was created on previous command line):
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the raw data folder:
DATA_PATH=${ROOT_PATH}/dosSantos_GenomicPrediction_2019

# Path of the asreml license:
ASREML_PATH=${ROOT_PATH}/asreml

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Run the code for processing the raw data:
${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/process_raw_data.py -dpath ${DATA_PATH} -rpath ${REPO_PATH} -opath ${OUT_PATH} & 

# Run the code for phenotypic data analysis:
Rscript ${REPO_PATH}/clean_repository/codes/phenotypic_data_analysis.R --asremlpath ${ASREML_PATH} --opath ${OUT_PATH} & 

# Run the code to estimate the heritability:
Rscript ${REPO_PATH}/clean_repository/codes/heritability_phenotypic_data_analysis.R --asremlpath ${ASREML_PATH} --opath ${OUT_PATH} & 

# Copy heritability table to the repository:
cp ${OUT_PATH}/heritabilities/heritabilities.csv ${REPO_PATH}/clean_repository/tables/

# Run the code for processing the raw data:
${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/create_figures_phenotypic_data_analysis.py -rpath ${REPO_PATH} -opath ${OUT_PATH} & 


#--------------Files for 5-fold cross-validation and forward-chaining cross-validation schemes---------------#

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Run the code to split the data into different subsets for cross-validation:
${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/split_data_cross_validation.py -opath ${OUT_PATH} & 

# Transform the .sh file into executable:
chmod +755 ${REPO_PATH}/clean_repository/codes/create_list_files_names_cross-validation.sh

# Create list of the name of the files with the data for cross-validation:
${REPO_PATH}/clean_repository/codes/create_list_files_names_cross-validation.sh -o ${OUT_PATH}


#-------------------To perform 5-fold cross-validation using the Bayesian Network model----------------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis (each analysis will use 4 threads to run 4 Markov chains):
n_analysis=10

for i in $(seq 1 ${n_analysis}); do  

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/y_cv5f_bn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${OUT_PATH}/processed_data/x_cv5f_bn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='BN'

	# Get the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${model}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

	# Run the code:
	${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/bayesian_network_analysis.py -y ${y} -x ${x} -model ${model} -rpath ${REPO_PATH} -opath ${OUT_PATH} -cvpath ${CV_OUT_PATH} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#---------------To perform forward-chaining cross-validation using the Bayesian Network model----------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis (each analysis will use 4 threads to run 4 Markov chains):
n_analysis=6

for i in $(seq 1 ${n_analysis}); do  

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/y_fcv_bn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${OUT_PATH}/processed_data/x_fcv_bn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='BN'

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${model}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

	# Run the code:
	${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/bayesian_network_analysis.py -y ${y} -x ${x} -model ${model} -rpath ${REPO_PATH} -opath ${OUT_PATH} -cvpath ${CV_OUT_PATH} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#--------------To perform 5-fold cross-validation using the Pleiotropic Bayesian Network model---------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis (each analysis will use 4 threads to run 4 Markov chains):
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/y_cv5f_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${OUT_PATH}/processed_data/x_cv5f_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
	model='PBN'

	# Get the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'_' -f2 <<<"$y")"
	tmp2="$(cut -d'-' -f1 <<<"$y")"
	tmp2="$(cut -d'_' -f3 <<<"$tmp2")"
	tmp3="$(cut -d'&' -f2 <<<"$y")"
	tmp3="$(cut -d'_' -f3 <<<"$tmp3")"

	# Get the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Create the name of the output directory:
	CV_OUT_PATH=${OUT_PATH}/cv/${model}/${tmp1}/${tmp2}-${tmp3}/${cv}

	# Run the code:
	${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/bayesian_network_analysis.py -y ${y} -x ${x} -model ${model} -rpath ${REPO_PATH} -opath ${OUT_PATH} -cvpath ${CV_OUT_PATH} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#---------To perform forward-chaining cross-validation using the Pleiotropic Bayesian Network model----------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis (each analysis will use 4 threads to run 4 Markov chains):
n_analysis=6

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/y_fcv_pbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${OUT_PATH}/processed_data/x_fcv_pbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
	model='PBN'

	# Get the name of the cross-validation scheme and traits:
	tmp1="$(cut -d'_' -f5 <<<"$y")"
	tmp2="$(cut -d'-' -f1 <<<"$y")"
	tmp2="$(cut -d'_' -f3 <<<"$tmp2")"
	tmp3="$(cut -d'_' -f6 <<<"$y")"

	# Get the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Create the name of the output directory:
	CV_OUT_PATH=${OUT_PATH}/cv/${model}/${tmp1}/${tmp2}-${tmp3}

	# Run the code:
	${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/bayesian_network_analysis.py -y ${y} -x ${x} -model ${model} -rpath ${REPO_PATH} -opath ${OUT_PATH} -cvpath ${CV_OUT_PATH} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#----------------To perform 5-fold cross-validation using the Dynamic Bayesian Network model-----------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis (each analysis will use 4 threads to run 4 Markov chains):
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/y_cv5f_dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${OUT_PATH}/processed_data/x_cv5f_dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
	model='DBN-0~6'

	# Get just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Get the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${tmp}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

	# Run the code:
	${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/bayesian_network_analysis.py -y ${y} -x ${x} -model ${model} -rpath ${REPO_PATH} -opath ${OUT_PATH} -cvpath ${CV_OUT_PATH} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-----------To perform forward-chaining cross-validation using the Dynamic Bayesian Network model------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis (each analysis will use 4 threads to run 4 Markov chains):
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/y_fcv_dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${OUT_PATH}/processed_data/x_fcv_dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
	model=$(sed -n "${i}p" ${OUT_PATH}/processed_data/dbn_models_fcv_list.txt)

	# Get just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${tmp}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

	# Run the code:
	${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/bayesian_network_analysis.py -y ${y} -x ${x} -model ${model} -rpath ${REPO_PATH} -opath ${OUT_PATH} -cvpath ${CV_OUT_PATH} & 

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#---------------To perform 5-fold cross-validation using the Multi Time GBLUP (MTi-GBLUP)--------------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/cv5f_mti-gblup_files.txt)

	# Name of the model: Multivariate Linear Mixed (MLM) model
	model="MTi-GBLUP-0~6"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")-$(cut -d'-' -f2 <<<"$model")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${tmp}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

	# Run the code:
	Rscript ${REPO_PATH}/clean_repository/codes/mixed_model_analysis.R --y ${y} --model ${model} --opath ${OUT_PATH} --cvpath ${CV_OUT_PATH} &

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-----------To perform forward-chaining cross-validation using the Multi Time GBLUP (MTi-GBLUP)--------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/fcv_mti-gblup_files.txt)

	# Name of the model: Multivariate Linear Mixed (MLM) model
	model=$(sed -n "${i}p" ${OUT_PATH}/processed_data/mti-gblup_models_fcv_list.txt)

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")-$(cut -d'-' -f2 <<<"$model")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${tmp}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

	# Run the code:
	Rscript ${REPO_PATH}/clean_repository/codes/mixed_model_analysis.R --y ${y} --model ${model} --opath ${OUT_PATH} --cvpath ${CV_OUT_PATH} &

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#----------------To perform 5-fold cross-validation using the Multi Trait GBLUP (MTr-GBLUP)------------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/cv5f_mtr-gblup_files.txt)

	# Name of the model: Multivariate Linear Mixed (MLM) model
	model="MTr-GBLUP-0~6"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")-$(cut -d'-' -f2 <<<"$model")"

	# Traits:
	traits="$(cut -d'_' -f3 <<<"$y")"-"$(cut -d'_' -f7 <<<"$y")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${tmp}/"$(cut -d'_' -f2 <<<"$y")"/${traits}/${cv}

	# Run the code:
	Rscript ${REPO_PATH}/clean_repository/codes/mixed_model_analysis.R --y ${y} --model ${model} --opath ${OUT_PATH} --cvpath ${CV_OUT_PATH} &

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#----------To perform forward-chaining cross-validation using the Multi Trait GBLUP (MTr-GBLUP)--------------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${OUT_PATH}/processed_data/fcv_mtr-gblup_files.txt)

	# Name of the model: Multivariate Linear Mixed (MLM) model
	model=$(sed -n "${i}p" ${OUT_PATH}/processed_data/mtr-gblup_models_fcv_list.txt)

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")-$(cut -d'-' -f2 <<<"$model")"

	# Traits:
	traits="$(cut -d'_' -f3 <<<"$y")"-"$(cut -d'_' -f6 <<<"$y")"

	# Define the output directory for the outputs:
	CV_OUT_PATH=${OUT_PATH}/cv/${tmp}/"$(cut -d'_' -f5 <<<"$y")"/${traits}

	# Run the code:
	Rscript ${REPO_PATH}/clean_repository/codes/mixed_model_analysis.R --y ${y} --model ${model} --opath ${OUT_PATH} --cvpath ${CV_OUT_PATH} &

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#--------To process results from the multivariate linear mixed models fitted with the EMMREML package--------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# To process results from the multivariate linear mixed models fitted with the EMMREML package:
Rscript ${REPO_PATH}/clean_repository/codes/process_emmreml_output.R --rpath ${REPO_PATH} --opath ${OUT_PATH} &


#---------To obtain results from the genomic prediction analysis under the cross-validation schemes----------#

# Set the root directory:
ROOT_PATH=/workdir/jp2476

# Set python path:
PYTHON_PATH=${ROOT_PATH}/local/python

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# To obtain the final results from the genomic prediction analysis:
${PYTHON_PATH}/bin/python ${REPO_PATH}/clean_repository/codes/genomic_prediction_results.py -rpath ${REPO_PATH} -opath ${OUT_PATH} & 

