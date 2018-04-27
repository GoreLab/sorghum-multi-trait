# General information

The repository contains codes and information related to the **sorghum multi trait project**. The data set has height measures over different days after planting, dry mass content, and other biomass compositional traits. The main goal of the project is the development of a **Bayesian Network** and a **Random Deep Neural Networks** that **integrates crop, physiological, genomic, and transcriptomic data**, and **recovery of information across traits** though the inclusion of **pleiotropic effects** into a unique model (Bayesian Network architecture). The benchmarking will be done via multi trait linear mixed models. A user friendly bash script will be written for massive run of deep neural networks through random guesses of hyperparameters in a multidimensional surface using **TensorFlow**, and the Bayesian Network will be written in the probabilistic programming language **stan**. The function should have as arguments the data frame, number of cores, and numbers of serial runs per core, everything automated for easy run in the linux terminal. The script will outputs rmse errors of the train, dev, and tests for all models evaluated, as well as plots for visualization of the results.

# Folder 'codes'

The folder 'codes' contains the codes written during the development of the project. The most important codes are:

1. **mtrait_code.py**: Contains all the python code for data processing, cleaning, unification of all data sets into a unique data frame, and the Bayesian Network and Random Deep Neural Network codes, as well as the cross-validation evaluation (under development).

2. **mtrait_data_processing.py**: Piece of the code within the 'mtrait_code.py', but just containing the data processing part (done).

3. **mtrait_cross_validation_and_models.py**: Piece of the code within the 'mtrait_code.py', but just containing the cross-validation and models evaluated (under development).

2. **external_functions.py**: Set of functions already written that the codes (1, 2, 3) depends on (done).

3. **mtrait_gdrive.sh**: Bash code for transfering data and plots between machines, and also as cloud backup mecanism (done).

4. **gbs_from_rdata_to_csv.R**: Code for transforming the raw ".RData" provided by Sam into ".csv" extension for easy loading into python (done).

5. **useful_bash_code.sh**: Useful unix-based bash code, including for run python processes into terminal, as well as to do some local software installation (under development).

6. **tmp.py**: Just to write temporary code in python.

7. **tmp.R**: Just to write temporary code in R.

# Folder 'notes'

Just for saving less important files, and also so save notes related to the project.

# Folder 'sublime'

Just for saving the workspace and project from sublime text editor

# Directories at the 'cbsugore02' lab cluster

**The github repository directory is (code, notes, and sublime workspace):** */workdir/jp2476/repo/sorghum-multi-trait*

**The gdrive repository directory is (data and plots):** */workdir/jp2476/repo/resul_mtrait-proj*

