# MTI Protein Imputation

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [License](https://github.com/goeckslab/MTIProteinImputation/blob/main/LICENSE)

## Overview

Multiplex tissue imaging are a collection of increasingly popular single-cell spatial proteomics and transcriptomics
assays for characterizing biological tissues both compositionally and spatially. However, several technical issues limit
the utility of multiplex tissue imaging, including the limited number of RNAs and proteins that can be assayed, tissue
loss, and protein probe failure. In this work, we demonstrate how machine learning methods can address these limitations
by imputing protein abundance at the single-cell level using multiplex tissue imaging datasets from a breast cancer
cohort. We first compared machine learning methods’ strengths and weaknesses for imputing single-cell protein abundance.
Machine learning methods used in this work include regularized linear regression, gradient-boosted regression trees, and
deep learning autoencoders. We also incorporated cellular spatial information to improve imputation performance. Using
machine learning, single-cell protein expression can be imputed with mean absolute error ranging between 0.05-0.3 on
a [0,1] scale. Our results demon-strate (1) the feasibility of imputing single-cell abundance levels for many proteins
using machine learning to overcome the technical constraints of multiplex tissue imaging and (2) how including cellular
spatial information can substantially enhance imputation results.

## Repo Contents

- [src](src): python source code for all analyses
- [data](data): Data folder used to store the data used in this study
- [figures](figures): Figures and tables generated by this study
- [results](results): Results of all experiments

## System Requirements

- Python 3.9 or higher
- RAM: 32GB or higher
- CPU: 8 cores or higher

Hardware Note:  
This research was performed and tested on an Intel platform and a Redhat platform.
Software library installation may require manual intervention on other hardware platforms, especially M1/M2/M3 Macs.

## Installation & Usage

### Step 1: Create virtual environment and install software libraries

- Use ```venv``` or ```conda``` to create a virtual environment that includes python 3.9
- E.g. ```python3 -m venv venv``` or ```conda create -n mti python=3.9```
- Activate the virtual environment
- Install needed software libraries: ```pip install -r requirements.txt```

### Step 2: Run Analyses (IMPORTANT NOTE: this will take a very long time)

Run this script from the root dir to execute all analyses and plot results:  
```./run_research.sh```

This script will run all experiments and scripts in order as well as generating the figures.

**IMPORTANT NOTE**:  
To replicate all results and findings **ALL** scripts have to be executed in order.
We recommend executing every experiment >=30 times to achieve statistical significance. Executing every experiment >=30
times can take >2
weeks.

If you want to execute only parts of the research use the scripts below.

#### Step 2a: Data Preparation

To create all spatial features, run the following script from the root dir:
```./src/data_preparation/download_data.sh```  
```./src/data_preparation/prepare_spatial_data.sh```

#### Step 2b: Run Analyses

##### Step 2b.1 Null Model
To run the Null Model experiments, run the following script from the root dir:
```./src/null_model/run_experiments.sh```

##### Step 2b.2: Elastic Net

To run the Elastic Net experiments, run the following script from the root dir:
```./src/en/run_experiments.sh <num_iterations>```

E.g.  
```./src/en/run_experiments.sh 30 ```
This will run the experiments 30 times.

##### Step 2b.3: LGBM

To run the LGBM experiments, run the following script from the root dir:
```./src/lgbm/run_experiments.sh <num_iterations>```

##### Step 2b.4: Auto Encoder

To run the Auto Encoder experiments, run the following script from the root dir:
```./src/ae/run_experiments.sh <num_iterations>```

##### Step 2b.5: Run cleanup script
```./src/cleanup/clean_score_datasets.sh```

#### Step 2b.6: Create required supplemental material
```./src/data_preparation/create_ae_supplemental_files.sh```

#### Step 2b.7: Run downstream classification analysis
```./src/classifier/run_downstream_classification.sh```

#### Step 2b.8: Run downstream clustering analysis
```python3 ./src/evaluation/cluster_evaluation.py```

#### Step 2b.9: Run phenotype analysis
```./src/phenotyping/run_experiments.sh```

##### Step 2b.10: Plotting

To create all figures and table as well as supplemental material, run the following script from the root dir:
```./src/figures/create_figures.sh```

**Attention**

To create the figures of the manuscript, **ALL** experiments have to be executed first.
We recommend to execute every experiment at least 30x to achieve statistical significance.

## Results

In this [preprint](https://www.biorxiv.org/content/10.1101/2023.12.05.570058v2) we show that it is possible to impute
protein abundance levels in multiplex tissue imaging data using machine learning. We also show that including spatial
information can improve imputation performance. We compare three different machine learning methods: Elastic Net,
Light Gradient Boosting Machines, and Autoencoders. While all three methods perform well, Autoencoders offer the
ability to impute multiple proteins at once, providing a distinct advantage over the other two methods.


## Citation
[![DOI](https://zenodo.org/badge/697920693.svg)](https://doi.org/10.5281/zenodo.14876654)