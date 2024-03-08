# Sparse Partial Bayesian NN 
Create folders: "datasets" and "output" in the main dir

To train a partial Bayes model: 
1. Train model deterministically (set network_type to deterministic)
    1. ```python train.py --base_setup "config.json"```
    2. default is config.json within the main dir - if a config file is in another dir pass "dir/config_file_name.json"
2. Train partial Bayes model by changing the following in the config file:
    1. Set network_type to "partial_bayesian"
    2. Set "init_network" in Partial Bayesian Parameters to the path to your trained deterministic model, e.g., "run_20240202_101010"

#### Used Packages: 
- Python > 3.8
- Pytorch > 2.0.1
- tqdm 4.65.0
- scikit-learn 1.2.2
- medMNIST
- h5py

#### System Requirement:
- CUDA v. > 11.7 (to be compatible with Pytorch 2)

### Folder Tree:

    main
    ├── config.json
    ├── dataset.py
    ├── test_model.py
    ├── train.py
    ├── utils.py
    └── models
        ├── __init__.py
        ├── bayeslayers.py
        ├── deterministic.py
        ├── maskedtensor.py
        ├── model.py
        ├── resnet18.py
        └── unet.py
    └── helpers
        ├── isic_dataset.py
        └── lidc_dataset.py
    └── output
    └── datasets

Datasets should be placed in a dataset folder and its own dataloader added into the `dataset.py` file. Current dataloaders are implemented for chestMNIST, LIDC-IDRI. 

for LIDC-IDRI download the dataset from this [link](https://drive.google.com/file/d/1QAtsh6qUgopFx1LJs20gOO9v5NP6eBgI/view?usp=drive_link) into the dataset folder. For ISIC follow this [link](https://challenge.isic-archive.com/data/#2018) download the segmentation train, validation, and test, unzip them, and run `preprocess_data` from `isic_dataset.py` to generate h5 file for training, validation and test for faster dataloading. ChestMNIST data is automatically downloaded from the MedMNIST library.

`utils.py` includes general utilities and network utilities, and many of the special sparse functions required for running sparse NN training.

`train.py` main training loop, requires passing the config file (defaults to the current config file in the main dir). Running the following command:

```python
python train.py
```

will read the config file in `main/config.json`. For other file names - pass the following argument 
```python
python train.py --base_setup path_to_config/config.json
```
The model training loop will automatically generate a folder called `run_YYYYMMDD-HHMMSS` with the datetimestamp. If no date is required pass `--no_date`. All other configurations can be configured in the `config.json` file. See details below as explanation for the rold of each key. 

`test_model.py` test model -- requires passing the name of the run, e.g.:

```python
python test_model.py --run "run_YYYYMDD-HHMMSS"
```
---
### Metrics & Network Types:
- Deterministic-Classification: Loss=CE, Metric=Accuracy
- Deterministic-Segmentation: Loss=CE, Metric=Dice or IoU
- Bayesian-Classification: ELBO(withNLL), Metric=Accuracy, KL, Brier Score, Entropy
- Bayesian-Segmentation: ELBO(withNLL), Metric=Dice or IoU, KL, Brier Score, Entropy
- Partial Bayesian-Classification: ELBO(withNLL), Metric=Accuracy, KL, Brier Score, Entropy
- Partial Bayesian-Segmentation: ELBO(withNLL), Metric=Dice or IoU, KL, Brier Score, Entropy

## Configuration File Explanation

The `config.json` configuration file contains settings that define various aspects of your neural network and training process. Here's a breakdown of the sections and their respective options: 

Note: the config file is saved for each run in the run folder

Sample config files available in configs/ folder

```
"Network": {
    "Basic Setup": {
        "network_type": "deterministic", #bayesian, partial_bayesian
        "architecture": "resnet", #MLP, cnn
        "layer_type": "conv", #linear
        "task": "classification", #regression
        "input_size": 784,
        "feature_size": 200,
        "output_size": 10, 
        "kernel_size": 3, 
        "padding_size": 1,
        "pooling_size": 0, 
        "stride": 2,
        "nonlinearity": "relu", #activation
        "init_weights": "uniform", #normal
        "use_batchnorm": false, 
        "bias": false #add bias or not
        },
    "Optional": {
        Sparsity: {
        "sparse": true, #to run sparse version of model -
        "random": false, #whether to randomly generate a sparse mask
        "rate": 0.1 #rate of sparsity 0 < r < 1
        }
    }
},
"Training": {
    "Basic Setup": {
        "init_network": "", #if continue_run == true, need to pass folder for init_network
        "continue_run": false, #for interrupted runs pass --continue_run flag
        "dataset": "isic", 
        "num_points": 
    },
    "Basic Parameters": {
        "num_epochs": 50,
        "optimizer": "sgd", #only use sgd for partial Bayes 
        "lr": 0.01, 
        "weight_decay": 1e-5, #for L2 reg
        "num_workers": 0,
        "batch_size": 50,
        "run_val": false, #whether running validation/test during training 
        "train_val_split": 0.9,
        "kfold": 1, 
        "seed": 0, #random seed for reproducibility
        "loss_func": "ce" 
    },
    "LR Scheduler":{
        "scheduler_type": "step",
        "step_size": 30,
        "gamma": 0.1
    },
    "Bayesian Parameters": {
        "kl_weight": 0.001, #weight of the kl term other options "epoch", "blundell" see code for more details
        "kl_weight_hi2low": true, #when selecting "epoch" whether to increase or decrease the weight as a function of epochs
        "kl_weight_lim": [0.01, 0.2], #defines the lower and upper limit of KL weight term
        "prior_mu": 0,  
        "prior_variance": 1,  
        "posterior_mu": 0, #min and max for initialization (for uniform dist -- if normal init is selected, input a single value)
        "posterior_rho": -3, 
        "num_samples": 5 #number of samples to sample from parameter distributions
    },
    "Partial Bayesian Parameters": {
        "init_network": "", #folder name in the output directory where the deterministic parameters are loaded
        "fit_variance_only": false, #wheter to fit the variance only when training partial bayesian network
        "bayesian_rate": 0.1, #rate of bayesian weights 0 < bayesian_rate < 1
        "largest_grad": true, #whether to select the largest gradients in sensitivity analysis or lowest (false)
        "node_analysis": "sensitivity_analysis", #only method implemented
        "select_random_node": false #whether to randomly select bayesian weights
        }
    },
    "Paths": {
            "dataset_path": "", #if different than default
            "training_path": "",
            "save_path": "" #if different than default
        }
```
