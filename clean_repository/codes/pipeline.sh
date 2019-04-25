

#-----------------------------------------Install python packages--------------------------------------------#

# Install python packages:
#--> Obs: Some packages may already be installed in your native python
pip3 install os
pip3 install argparse
pip3 install pandas
pip3 install numpy

# Install R packages:
Rscript -e 'install.packages("optparse", repos="https://cran.rstudio.com")'
Rscript -e 'install.packages("data.table", repos="https://cran.rstudio.com")'
Rscript -e 'install.packages("magrittr", repos="https://cran.rstudio.com")'
Rscript -e 'install.packages("lme4", repos="https://cran.rstudio.com")'
Rscript -e 'install.packages("sjstats", repos="https://cran.rstudio.com")'

#----------------------------------Define the path of repository folder--------------------------------------#

# Prefix of the output directory is:
ROOT_PATH=/home/jhonathan/Documents

#------------------------------------------Create directories------------------------------------------------#

# Path of the repository:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Transform the .sh file into executable:
chmod +755 ${REPO_PATH}/clean_repository/codes/create_directories.sh 

# Create directories:
${REPO_PATH}/clean_repository/codes/create_directories.sh -p ${ROOT_PATH}

	
#-----------------------------------------Process the raw data-----------------------------------------------#

# Path with the raw data downloaded from:
DATA_PATH=${ROOT_PATH}/raw_data_sorghum-multi-trait

# Path of the repository:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Run the code for processing the raw datas:
python3 ${REPO_PATH}/clean_repository/codes/process_raw_data.py -dpath ${DATA_PATH} -rpath ${REPO_PATH} -opath ${OUT_PATH} & 


#---------------------------------------Phenotypic data analysis-----------------------------------------------#

# Path of the repository:
REPO_PATH=${ROOT_PATH}/sorghum-multi-trait

# Path of the output folder:
OUT_PATH=${ROOT_PATH}/output_sorghum-multi-trait

# Run the code for phenotypic data analysis:
Rscript ${REPO_PATH}/clean_repository/codes/phenotypic_data_analysis.R --rpath ${REPO_PATH} --opath ${OUT_PATH} & 

