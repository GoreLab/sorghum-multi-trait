
#---------------------------------------Define the path of the folders---------------------------------------#

# Prefix of the output directory is:
ROOT_PATH=/home/jhonathan/Documents
# ROOT_PATH=/workdir/jp2476/repo

# Path of the raw data folder:
DATA_PATH=${ROOT_PATH}/raw_data_sorghum-multi-trait

# Path of the repository folder:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait


#--------------------------Install python locally (if you do not have installed or face bugs)----------------#

# Create a path to instal the softweres locally:
mkdir ${ROOT_PATH}/local_install_sorghum-multi-trait/
cd ${ROOT_PATH}/local_install_sorghum-multi-trait/

# Install python locally (please select a path to install python to be informed forward):
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh

# Set the path where python was locally installed until the bin directoy (this is a example):
PYTHON_PATH=${ROOT_PATH}/local_install_sorghum-multi-trait/python/bin


#-----------------------------------------Install python packages--------------------------------------------#

# Install python packages:
pip3 install argparse==1.4.0
pip3 install numpy==1.16.3
pip3 install pandas==0.24.2
pip3 install sklearn==0.0
pip3 install matplotlib==3.0.3
pip3 install seaborn==0.9.0

# Install R packages:
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/optparse/optparse_1.6.0.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/data.table/data.table_1.12.0.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/magrittr/magrittr_1.0.1.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/lme4/lme4_1.1-18-1.tar.gz", repos=NULL, type="source")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/stringr/stringr_1.3.1.tar.gz", repos=NULL, type="source")'


#------------------------------------------Create directories------------------------------------------------#

# Transform the .sh file into executable:
chmod +755 ${REPO_PATH}/clean_repository/codes/create_directories.sh 

# Create directories:
${REPO_PATH}/clean_repository/codes/create_directories.sh -p ${ROOT_PATH}


#---------------------------------------Phenotypic data analysis---------------------------------------------#

# Run the code for processing the raw data:
python3 ${REPO_PATH}/clean_repository/codes/process_raw_data.py -dpath ${DATA_PATH} -rpath ${REPO_PATH} -opath ${OUT_PATH} & 

# Run the code for phenotypic data analysis:
Rscript ${REPO_PATH}/clean_repository/codes/phenotypic_data_analysis.R --opath ${OUT_PATH} & 

# # Run the code to estimate the heritability:
# Rscript ${PREFIX_code}/mtrait_first_step_analysis_heritability_std_delta.R & 

# Run the code for processing the raw data:
python3 ${REPO_PATH}/clean_repository/codes/create_figures_phenotypic_data_analysis.py -rpath ${REPO_PATH} -opath ${OUT_PATH} & 


#--------Files for 5-fold cross-validation (cv5f) and forward-chaining cross-validation (fcv) schemes--------#

# Run the code to split the data into different subsets for cross-validation:
python3 ${REPO_PATH}/clean_repository/codes/split_data_cross_validation.py -opath ${OUT_PATH} & 

# Transform the .sh file into executable:
chmod +755 ${REPO_PATH}/clean_repository/codes/create_list_files_names_cross-validation.sh

# Create list of the name of the files with the data for cross-validation:
${REPO_PATH}/clean_repository/codes/create_list_files_names_cross-validation.sh -o ${OUT_PATH}
