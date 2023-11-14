# MTI Protein Imputation

# Step 1: Create virtual environment and install software libraries

- Use ```venv``` or ```conda``` to create a virtual environment that includes python 3.9 
- E.g. ```python3 -m venv venv``` or ```conda create -n mti python=3.9```
- Activate the virtual environment
- Install needed software libraries: ```pip install -r requirements.txt```

### Hardware Note:
This research was performed and tested on an Intel platform and a Redhat platform. Software library installation may require manual intervention on other hardware platforms, especially M1/M2 Macs.

# Step 2: Run Analyses (IMPORTANT NOTE: this will take a very long time)

Run this script from the root dir to execute all analyses and plot results:  
```./run_research.sh```

This script will run all experiments and scripts in order as well as generating the figures.  

**IMPORTANT NOTE**:  
To replicate all results and findings **ALL** scripts have to be executed in order.
We recommend executing every experiment >=30 times to achieve statistical significance. Executing every experiment >=30 times can take >2
weeks.

If you want to execute only parts of the research use the scripts below.

## Step 2a: Data Preparation

To create all spatial features, run the following script from the root dir:
```./src/data_preparation/prepare_spatial_data.sh```

## Step 2b: Run Analyses

### Step 2b.1: Elastic Net

To run the Elastic Net experiments, run the following script from the root dir:
```./src/en/run_experiments.sh <num_iterations>```

E.g.  
```./src/en/run_experiments.sh 30 ```
This will run the experiments 30 times.

### Step 2b.2: LGBM

To run the LGBM experiments, run the following script from the root dir:
```./src/lgbm/run_experiments.sh <num_iterations>```

### Step 2b.3: Auto Encoder

To run the Auto Encoder experiments, run the following script from the root dir:
```./src/ae/run_experiments.sh <num_iterations>```

# Step 2b.4: Plotting

To create all figures and table as well as supplemental material, run the following script from the root dir:
```./src/figures/create_figures.sh```

### Attention

To create the figures of the manuscript, **ALL** experiments have to be executed first.
We recommend to execute every experiment at least 30x to achieve statistical significance.





