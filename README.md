# General information

The repository contains codes and information related to the sorghum multi trait project. The data set has height measures over different days after planting, dry mass content, and other biomass compositional traits. The main goal of the project is the development of a **Bayesian Network** and a **Random Deep Neural Networks** that **integrates crop, physiological, genomic, and transcriptomic data**, and **recovery of information** across traits though the inclusion of **pleiotropic effects** into a unique model (Bayesian Network architecture). The benchmarking will be done via multi trait linear mixed models. A code also will be written for massive run of deep neural networks through random guesses of hyperparameters in a multivariate surface using TensorFlow, and the Bayesian Network will be written in the probabilistic programming language stan.
# Folder 'codes'

The folder 'codes' contains the codes written during the development of the project. The most important codes are:

1. mtrait_code.py: Contains all the python code for data processing, cleaning, unification of all data sets into a unique data frame, and the Bayesian Network and Random Deep Neural Network codes, as well as the cross-validation evaluation.

2. external_functions.py: Set of functions already written that the code 'mtrait_code.py' depends on.

3. mtrait_gdrive.sh: Bash code for transfering data and plots between machines, and also as cloud backup mecanism, and other useful bash codes.

4. gbs_from_rdata_to_csv.R: Code for transforming the raw ".RData" provided by Sam into ".csv" extension for easy loading into python.

5. tmp.py: Just to write temporary code in python.

6. tmp.R: Just to write temporary code in R.

# Folder 'notes'

Just for saving less important files, and also so save notes related to the project

# Folder 'sublime'

Just for saving the workspace and project from sublime text editor





