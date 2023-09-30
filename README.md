# MTIProteinImputation

# Installation

1. ```Create a virtual environment using python 3.9```
2. ```Activate venv```
3. ```pip install -r requirements.txt```

# Usage

Disclaimer:  
To replicate all results and findings **ALL** scripts have to be executed in order.
Additionally, experiments for the Elastic Net, LGBM and Auto Encoder have to be repeated at least 30x, which can take 2 weeks plus.


## Data Preparation

To create all spatial features, run the following script from the root dir:
```./src/data_preparation/prepare_spatial_data.sh```


## Experiments

### Elastic Net
To run the Elastic Net experiments, run the following script from the root dir:
```./src/experiments/elastic_net/run_experiments.sh```

To specify the number of repetitions, 
add the number at the end of the script as an additional parameter.

E.g.  
```./src/experiments/elastic_net/run_experiments.sh 30 ```
This will run the experiments 30 times.

### LGBM
To run the LGBM experiments, run the following script from the root dir:
```./src/experiments/lgbm/run_experiments.sh```


### Auto Encoder
To run the Auto Encoder experiments, run the following script from the root dir:
```./src/experiments/auto_encoder/run_experiments.sh```


# Plotting

To create the figures of the manuscript, **ALL** experiments have to be executed first.
We recommend to execute every experiment at least 30x, which can take 2 weeks plus.

