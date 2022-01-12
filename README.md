# Model-Contrastive Federated Learning
This repository is a reproducibility study of [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html) (or for short, MOON).

## Setup

Installation requirements:
- `poetry`
- `python > 3.7`

**Poetry**
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

To configure current shell:
```bash
source $HOME/.poetry/env
```

Go into project directory, and install dependencies:
```bash
cd MOON/
poetry install
```

Activate virtual environment created by poetry:
```bash
poetry shell
```

## Run experiments

We have provided shell scripts that run the experiments. They are in the `run_scripts/` directory.
For example, to run the basic experiment (comparing FedAvg and Moon accuracies on 10 clients, 100 rounds):
```bash
./run_experiments/basic_experiment.sh
```


## Parameters

| Parameter           	 | Description                                                                                                                                                                                                                                                                              	|
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| num_rounds          	 | Number of rounds of federated learning (default: 1)                                                                                                                                                                                                                                      	|
| sample_fraction_fit 	 | Fraction of available clients used for fit (default: 1.0)                                                                                                                                                                                                                                	|
| num_clients         	 | Number of clients to start (default: 100)                                                                                                                                                                                                                                                	|
| strategy            	 | Name of the strategy you are running (default: fedAvg). Alternatively, use moon.                                                                                                                                                                                                         	|
| dataset             	 | Dataset to train on. Currently support only CIFAR-10.                                                                                                                                                                                                                                    	|
| beta               	 | Concentration parameter of Dirichlet Distribution (default: 0.5)                                                                                                                                                                                                                         	|
| mu                  	 | Mu for MOON (default: 5)                                                                                                                                                                                                                                                                 	|
| seed                	 | Initial seed                                                                                                                                                                                                                                                                             	|
