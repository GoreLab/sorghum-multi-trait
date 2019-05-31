#  **Novel Bayesian Networks for Genomic Prediction of Developmental Traits in Biomass Sorghum**
![language: R](https://img.shields.io/badge/language-R-blue.svg)
![language: Python](https://img.shields.io/badge/language-Python-green.svg)
![status: WIP](https://img.shields.io/badge/status-WorkInProgress-red.svg)

![alt text](https://github.com/GoreLab/sorghum-multi-trait/blob/master/edited_figures/figure_1_bayesian_networks_figures/inkscape/nets.png)

# **Abstract**

The ability to connect information between traits over time allow Bayesian networks to offer a powerful probabilistic framework to construct genomic prediction models. In this study, we phenotyped a diversity panel of 869 biomass sorghum lines, which had been genotyped with 100,435 SNP markers, for plant height (PH) with biweekly measurements from 30 to 120 days after planting (DAP) and for end-of-season dry biomass yield (DBY) in four environments. We developed and evaluated five genomic prediction models: Bayesian network (BN), Pleiotropic Bayesian network (PBN), Dynamic Bayesian network (DBN), multi-trait GBLUP (MTr-GBLUP), and multi-time GBLUP (MTi-GBLUP) models. In 5-fold cross-validation, prediction accuracies ranged from 0.48 (PBN) to 0.51 (MTr-GBLUP) for DBY and from 0.47 (DBN, DAP120) to 0.74 (MTi-GBLUP, DAP60) for PH. Forward-chaining cross-validation further improved prediction accuracies (42-51\%) of the DBN, MTi-GBLUP and MTr-GBLUP models for PH (train on 45 DAP, predict 120 DAP). Coincidence indices (target: biomass, secondary: PH) and a coincidence index based on lines (PH time series) showed that the ranking of lines by PH changed minimally after 45 DAP. These results suggest a two-level indirect selection method for PH at harvest (first-level target trait) could be performed earlier in the season based on ranking of lines by PH at 45 DAP (secondary trait) and DBY (second-level target trait). With the advance of high-throughput phenotyping technologies, statistical approaches such as our proposed two-level indirect selection framework could be valuable for enhancing genetic gain per unit of time when selecting on developmental traits.

# **Guidelines**

# Folder 'codes'

The folder 'codes' contains the codes written during the development of the project. The most important codes are:

1. **mtrait_code.py**: Contains all the python code for data processing, cleaning, unification of all data sets into a unique data frame, and the Bayesian Network and Random Deep Neural Network codes, as well as the cross-validation evaluation (under development).

2. **mtrait_data_processing.py**: Piece of the code within the 'mtrait_code.py', but just containing the data processing part (done).

3. **mtrait_cross_validation_and_models.py**: Piece of the code within the 'mtrait_code.py', but just containing the cross-validation and models evaluated (under development).

4. **mtrait_iter_chain_number_tst.py**: Used for tuning the number of iterations and chains to fit the Bayesian Networks (done).

5. **multi_trait.stan**: Code written at the probabilistic programming language stan to fit the Bayesian Network without pleiotropic effects (done).

6. **plots_raw_data.R**: For plotting different types of plots to explore the raw data features (done).

7. **external_functions.py**: Set of functions already written that the codes (1, 2, 3) depends on (done).

8. **mtrait_gdrive.sh**: Bash code for transfering data and plots between machines, and also as cloud backup mechanism (done).

9. **gbs_from_rdata_to_csv.R**: Code for transforming the raw ".RData" provided by Sam into ".csv" extension for easy loading into python (done).

10. **useful_bash_code.sh**: Useful unix-based bash code, including for run python processes into terminal, as well as to do some local software installation (under development).

11. **tmp.py**: Just to write temporary code in python.

12. **tmp.R**: Just to write temporary code in R.

# Folder 'notes'

Just for saving less important files, and also so save notes related to the project.

# Folder 'sublime'

Just for saving the workspace and project from sublime text editor

# Directories at the 'cbsugore02' lab cluster

**The github repository directory is (code, notes, and sublime workspace):** */workdir/jp2476/repo/sorghum-multi-trait*

**The gdrive repository directory is (data and plots):** */workdir/jp2476/repo/resul_mtrait-proj*

