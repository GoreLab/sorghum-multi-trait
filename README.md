#  **Novel Bayesian Networks for Genomic Prediction of Developmental Traits in Biomass Sorghum**
![language: R](https://img.shields.io/badge/language-R-blue.svg)
![language: Python](https://img.shields.io/badge/language-Python-green.svg)
![status: WIP](https://img.shields.io/badge/status-WorkInProgress-red.svg)

<p align="center"><a href="https://github.com/GoreLab/sorghum-multi-trait/blob/master/edited_figures/figure_3_forward_cross_validation_figures/inkscape/heatplot_fcv_accuracy.png"><img src="https://github.com/GoreLab/sorghum-multi-trait/blob/master/edited_figures/figure_3_forward_cross_validation_figures/inkscape/heatplot_fcv_accuracy.png" width="555" height="328"/></a>

# **Abstract**

The ability to connect information between traits over time allow Bayesian networks to offer a powerful probabilistic framework to construct genomic prediction models. In this study, we phenotyped a diversity panel of 869 biomass sorghum lines, which had been genotyped with 100,435 SNP markers, for plant height (PH) with biweekly measurements from 30 to 120 days after planting (DAP) and for end-of-season dry biomass yield (DBY) in four environments. We developed and evaluated five genomic prediction models: Bayesian network (BN), Pleiotropic Bayesian network (PBN), Dynamic Bayesian network (DBN), multi-trait GBLUP (MTr-GBLUP), and multi-time GBLUP (MTi-GBLUP) models. In 5-fold cross-validation, prediction accuracies ranged from 0.48 (PBN) to 0.51 (MTr-GBLUP) for DBY and from 0.47 (DBN, DAP120) to 0.74 (MTi-GBLUP, DAP60) for PH. Forward-chaining cross-validation further improved prediction accuracies (36.4-52.4\%) of the DBN, MTi-GBLUP and MTr-GBLUP models for PH (training slice: 30-45 DAP). Coincidence indices (target: biomass, secondary: PH) and a coincidence index based on lines (PH time series) showed that the ranking of lines by PH changed minimally after 45 DAP. These results suggest a two-level indirect selection method for PH at harvest (first-level target trait) and DBY (second-level target trait) could be conducted earlier in the season based on ranking of lines by PH at 45 DAP (secondary trait). With the advance of high-throughput phenotyping technologies, statistical approaches such as our proposed two-level indirect selection framework could be valuable for enhancing genetic gain per unit of time when selecting on developmental traits.

# **Guidelines**

Instructions to reproduce the project, codes description, and data availability on the file `README.txt`.
