
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

# Set directory where the data is:
cd /workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation

# List training data files names related to the CV1 scheme and storing it for latter usage:
ls y*cv1*trn* > y_cv1_bn_trn_files.txt
ls x*cv1*trn* > x_cv1_bn_trn_files.txt
ls y*cv1*height*trn* > y_cv1_dbn_trn_files.txt
ls x*cv1*height*trn* > x_cv1_dbn_trn_files.txt

# List training files names related to the CV1 scheme and storing it for latter usage on PBN model
grep "drymass" y_cv1_bn_trn_files.txt > tmp1.txt 
grep "height" y_cv1_bn_trn_files.txt > tmp2.txt 
paste -d'&' tmp1.txt tmp2.txt > y_cv1_pbn_trn_files.txt
grep "drymass" x_cv1_bn_trn_files.txt > tmp1.txt 
grep "height" x_cv1_bn_trn_files.txt > tmp2.txt 
paste -d'&' tmp1.txt tmp2.txt > x_cv1_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# List training phenotypic data files names related to the CV2 scheme and storing it for latter usage (BN):
ls y*cv2*only*trn* > y_cv2_bn_trn_files.txt
ls x*cv2*only*trn* > x_cv2_bn_trn_files.txt

# List training phenotypic data files names related to the CV2 scheme and storing it for latter usage (PBN):
echo 'y_cv2_drymass_trn.csv' > tmp1.txt
echo 'x_cv2_drymass_trn.csv' > tmp2.txt
for i in $(seq 1 5); do 

	echo 'y_cv2_drymass_trn.csv' >> tmp1.txt
	echo 'x_cv2_drymass_trn.csv' >> tmp2.txt

done;
paste -d'&' tmp1.txt y_cv2_bn_trn_files.txt > y_cv2_pbn_trn_files.txt
paste -d'&' tmp2.txt x_cv2_bn_trn_files.txt > x_cv2_pbn_trn_files.txt
rm tmp1.txt tmp2.txt

# List training data files names related to the CV2 scheme and storing it for latter usage (DBN):
ls y*cv2*height*trn* > y_cv2_dbn_trn_files.txt
ls x*cv2*height*trn* > x_cv2_dbn_trn_files.txt
sed -i '/only/d' y_cv2_dbn_trn_files.txt
sed -i '/only/d' x_cv2_dbn_trn_files.txt

# Create a text file to store the different types of Dynamic Bayesian network models for latter usage (DBN);
echo "DBN-0~5" > dbn_models_cv2_list.txt
echo "DBN-0~1" >> dbn_models_cv2_list.txt
echo "DBN-0~2" >> dbn_models_cv2_list.txt
echo "DBN-0~3" >> dbn_models_cv2_list.txt
echo "DBN-0~4" >> dbn_models_cv2_list.txt

# List of files to get train data index into R to perform Multivariate Linear Mixed (MLM) model analysis in R:
cat y_cv1_dbn_trn_files.txt > cv1_mtilm_files.txt
cat y_cv1_pbn_trn_files.txt > cv1_mtrlm_files.txt
cat y_cv2_dbn_trn_files.txt > cv2_mtilm_files.txt
echo 'y_cv2_drymass_trn.csv' > tmp1.txt
for i in $(seq 1 4); do 

	echo 'y_cv2_drymass_trn.csv' >> tmp1.txt

done;
paste -d'&' tmp1.txt y_cv2_dbn_trn_files.txt > cv2_mtrlm_files.txt
rm tmp1.txt

# Create a text file to store the different types of Multi Time Linear Mixed (MTiLM) models for latter usage;
echo "MTiLM-0~5" > mtilm_models_cv2_list.txt
echo "MTiLM-0~1" >> mtilm_models_cv2_list.txt
echo "MTiLM-0~2" >> mtilm_models_cv2_list.txt
echo "MTiLM-0~3" >> mtilm_models_cv2_list.txt
echo "MTiLM-0~4" >> mtilm_models_cv2_list.txt

# Create a text file to store the different types of Multi Trait Linear Mixed (MTrLM) models for latter usage;
echo "MTrLM-0~5" > mtrlm_models_cv2_list.txt
echo "MTrLM-0~1" >> mtrlm_models_cv2_list.txt
echo "MTrLM-0~2" >> mtrlm_models_cv2_list.txt
echo "MTrLM-0~3" >> mtrlm_models_cv2_list.txt
echo "MTrLM-0~4" >> mtrlm_models_cv2_list.txt

# Create a text file to store the different types of Growth Bayesian Network (GBN) models for latter usage;
echo "GBN-0~5-cv2" > gbn_models_cv2_list.txt
echo "GBN-0~1-cv2" >> gbn_models_cv2_list.txt
echo "GBN-0~2-cv2" >> gbn_models_cv2_list.txt
echo "GBN-0~3-cv2" >> gbn_models_cv2_list.txt
echo "GBN-0~4-cv2" >> gbn_models_cv2_list.txt


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

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
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

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
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

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
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


#-------------To perform cross-validation (CV1) analysis using the Multi time Linear Mixed model-------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/cv1_mtilm_files.txt)

	# Name of the model:
	model="MTiLM-0~6"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

	# Define the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

	# Run the code:
	Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-------------To perform cross-validation (CV2) analysis using the Multi time Linear Mixed model-------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/cv2_mtilm_files.txt)

	# Name of the model that can be:
	model=$(sed -n "${i}p" ${dir_in}/mtilm_models_cv2_list.txt)

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

	# Defining the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

	# Run the code:
	Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#--------To perform cross-validation (CV1) analysis using the Multi trait time Linear Mixed model------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/cv1_mtrlm_files.txt)

	# Name of the model:
	model="MTrLM-0~6"

	# Getting the current fold of the cross-validation:
	cv="$(cut -d'_' -f4 <<<"$y")"

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

	# Get the trait names:
	tmp="$(cut -d'_' -f3 <<<"$y")"-"$(cut -d'_' -f7 <<<"$y")"

	# Define the output directory for the outputs:
	dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/${tmp}/${cv}

	# Run the code:
	Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#--------To perform cross-validation (CV2) analysis using the Multi trait time Linear Mixed model------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/cv2_mtrlm_files.txt)

	# Name of the model:
	model=$(sed -n "${i}p" ${dir_in}/mtrlm_models_cv2_list.txt)

	# Directory of the project folder:
	dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

	# Getting just the model prefix:
	tmp="$(cut -d'-' -f1 <<<"$model")"

	# Prefix of the output directory:
	PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

	# Get current cv2:
	cv="$(cut -d'_' -f5 <<<"$y")"

	# Get the trait names:
	tmp="$(cut -d'_' -f3 <<<"$y")"-"$(cut -d'_' -f6 <<<"$y")"

	# Define the output directory for the outputs:
	dir_out=${PREFIX}/${cv}/${tmp}

	# Run the code:
	Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

	# Sleep for avoid exploding several processes:
	sleep 5

done;


#-------------To perform cross-validation (CV1) analysis using the Growth Bayesian Network model------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv1_dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv1_dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
	model='GBN-0~6-cv1'

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

#-------------To perform cross-validation (CV2) analysis using the Growth Bayesian Network model------------#

# Number of analysis:
n_analysis=5

for i in $(seq 1 ${n_analysis}); do

	# Directory of the folder where y and x are stored:
	dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

	# Name of the file with the phenotypes:
	y=$(sed -n "${i}p" ${dir_in}/y_cv2_dbn_trn_files.txt)

	# Name of the file with the features:
	x=$(sed -n "${i}p" ${dir_in}/x_cv2_dbn_trn_files.txt)

	# Name of the model that can be: 'BN' or 'PBN', or 'DBN'
	model=$(sed -n "${i}p" ${dir_in}/gbn_models_cv2_list.txt)

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


############################# MTiLM model CV1 ###################################

i=5

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Name of the file with the phenotypes:
y=$(sed -n "${i}p" ${dir_in}/cv1_mtilm_files.txt)

# Name of the model: Multivariate Linear Mixed (MLM) model
model="MTiLM-0~6"

# Getting the current fold of the cross-validation:
cv="$(cut -d'_' -f4 <<<"$y")"

# Directory of the project folder:
dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

# Getting just the model prefix:
tmp="$(cut -d'-' -f1 <<<"$model")"

# Prefix of the output directory:
PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

# Define the output directory for the outputs:
dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"/${cv}

# Run the code:
Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

# Sleep for avoid exploding several processes:
sleep 5


############################# MTiLM model CV2 ###################################

i=5

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Name of the file with the phenotypes:
y=$(sed -n "${i}p" ${dir_in}/cv2_mtilm_files.txt)

# Name of the model that can be: 'BN' or 'PBN', or 'DBN':
model=$(sed -n "${i}p" ${dir_in}/mtilm_models_cv2_list.txt)

# Directory of the project folder:
dir_proj="/workdir/jp2476/repo/sorghum-multi-trait/"

# Getting just the model prefix:
tmp="$(cut -d'-' -f1 <<<"$model")"

# Prefix of the output directory:
PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

# Defining the output directory for the outputs:
dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/"$(cut -d'_' -f3 <<<"$y")"

# Run the code:
Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

# Sleep for avoid exploding several processes:
sleep 5


############################# MTrLM model CV1 ###################################

i=5

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Name of the file with the phenotypes:
y=$(sed -n "${i}p" ${dir_in}/cv1_mtrlm_files.txt)

# Name of the model: Multivariate Linear Mixed (MLM) model
model="MTrLM-0~6"

# Getting the current fold of the cross-validation:
cv="$(cut -d'_' -f4 <<<"$y")"

# Directory of the project folder:
dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

# Getting just the model prefix:
tmp="$(cut -d'-' -f1 <<<"$model")"

# Prefix of the output directory:
PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

# Get the trait names:
tmp="$(cut -d'_' -f3 <<<"$y")"-"$(cut -d'_' -f7 <<<"$y")"

# Define the output directory for the outputs:
dir_out=${PREFIX}/"$(cut -d'_' -f2 <<<"$y")"/${tmp}/${cv}

# Run the code:
Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

# Sleep for avoid exploding several processes:
sleep 5


############################# MTrLM model CV2 ###################################

i=1

# Directory of the folder where y and x are stored:
dir_in="/workdir/jp2476/repo/resul_mtrait-proj/data/cross_validation/"

# Name of the file with the phenotypes:
y=$(sed -n "${i}p" ${dir_in}/cv2_mtrlm_files.txt)

# Name of the model: Multivariate Linear Mixed (MLM) model
model=$(sed -n "${i}p" ${dir_in}/mtrlm_models_cv2_list.txt)

# Directory of the project folder:
dir_proj="/workdir/jp2476/repo/sorghum-multi-trait"

# Getting just the model prefix:
tmp="$(cut -d'-' -f1 <<<"$model")"

# Prefix of the output directory:
PREFIX="/workdir/jp2476/repo/resul_mtrait-proj/outputs/cross_validation/${tmp}"

# Get current cv2:
cv="$(cut -d'_' -f5 <<<"$y")"

# Get the trait names:
tmp="$(cut -d'_' -f3 <<<"$y")"-"$(cut -d'_' -f6 <<<"$y")"

# Define the output directory for the outputs:
dir_out=${PREFIX}/${cv}/${tmp}

# Run the code:
Rscript ${dir_proj}/codes/mtrait_mixed_models.R ${y} ${model} ${dir_in} ${dir_proj} ${dir_out};

# Sleep for avoid exploding several processes:
sleep 5

echo $y
echo $model
echo $dir_in
echo $dir_proj
echo $dir_out
