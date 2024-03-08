import argparse
import os, operator
from pathlib import Path
import numpy as np
import random, json
import dataset
import models.model as model
from sklearn.model_selection import KFold
import time
import torch
import torch.utils.data
from tqdm import tqdm, trange
import utils
from itertools import islice

"""======================================================================================="""
### ARGUMENTS
parse_in = argparse.ArgumentParser()
parse_in.add_argument('--base_setup',   type=str, default='config.json',
										help='Path to configuration file')
parse_in.add_argument('--search_setup', type=str, default='',
										help='Path to search for config file')
parse_in.add_argument('--no_date',      action='store_true', help='Do not use date when logging files.')
parse_in.add_argument('--continue_run', action='store_true', help='continue interrupted run')
parse_in.add_argument('--seed', 		type=int, default=None, help='random seed' )

opt = parse_in.parse_args()

assert opt.base_setup!='', 'Please provide a config file'

if opt.search_setup == '': 
	opt.search_setup = Path(__file__).resolve().parent

opts = utils.read_config(opt.search_setup, file=opt.base_setup)
if opts["Paths"]["save_path"] == '':
	opts["Paths"]["save_path"] = str(Path(__file__).resolve().parent / 'output' / 'run')
else:
	opts["Paths"]["save_path"] += '/run' 
if not opt.no_date:
	opts["Paths"]["save_path"] += '_' + time.strftime("%Y%m%d-%H%M%S")
	
if opts["Paths"]["dataset_path"] != '':
	dataset_path = opts["Paths"]["dataset_path"]
else:
	dataset_path = os.path.join(opt.search_setup, 'datasets')

### NETWORK SET UP 
tmp = opts["Network"]["Basic Setup"]
network_type = tmp["network_type"]
net_architecture = tmp["architecture"]
layer_type = tmp["layer_type"]
task = tmp["task"]
batch_norm = tmp["use_batchnorm"]  
init_weights= tmp["init_weights"]
nonlinearity = tmp["nonlinearity"]
input_size = tmp["input_size"]
feature_size = tmp["feature_size"]
output_size = tmp["output_size"]
kernel_size=tmp["kernel_size"]
padding_size=tmp["padding_size"]
pooling_size=tmp["pooling_size"]
stride=tmp["stride"]
bias = tmp["bias"]

tmp = opts["Training"]["Basic Setup"]
init_network = tmp["init_network"]
continue_run = tmp["continue_run"]
dataset_setup = tmp['dataset']
num_points = tmp["num_points"]

### OPTIONAL PARAMETERS
tmp = opts["Network"]["Optional"]
sparsity = tmp["Sparsity"]

### TRAINING PARAMETERS
tmp = opts["Training"]["Basic Parameters"]
optimizer_type = tmp["optimizer"]
lr = tmp["lr"]
batch_size = tmp["batch_size"]
kfold = tmp["kfold"]
cuda_id = tmp["gpu"]
run_val = tmp["run_val"] if kfold == 1 else True
train_val_split = tmp["train_val_split"]
num_epochs = tmp["num_epochs"]
weight_decay = tmp["weight_decay"]
loss_func = tmp["loss_func"]
nll_loss = True if "nll" in loss_func else False #needed to check if log_softmax is needed before loss calculation
multi_label = True if loss_func == "bceLogits" and output_size > 2 else False
criterion = utils.get_loss_func(loss_func, dataset=dataset_setup) 
init_params = None 
if opt.seed is not None:
	seed = opt.seed
	tmp['seed'] = seed
else:
	seed = tmp['seed']

### SET UP SEED FOR REPRODUCIBILITY
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')

"""======================================================================================="""
### LOAD DATA
flatten_input = True if layer_type == 'linear' and task == "classification" else False
data_loader, val_loader, opts["Training"]["Basic Parameters"]["batch_size"] = dataset.get_dataloader(root=dataset_path, dataset=dataset_setup,
																								kfold=kfold, input_size=input_size,
																								batch_size=batch_size, train_shuffle=True, 
																								train_split=train_val_split, run_val=run_val,
																								num_points=num_points, flatten_input=flatten_input)

"""======================================================================================="""
### NETWORK INITIALIZATION
tmp = opts["Network"]["Basic Setup"]

if network_type == "partial_bayesian":
	def initialize_partial_bayes(fold=None):
		partial_bayes_params = opts["Training"]["Partial Bayesian Parameters"]
		init_net = partial_bayes_params["init_network"]
		init_net_path = Path(opts["Paths"]["save_path"]).parent / init_net
		if not init_net_path.exists(): 
			output_directory = next((p for p in reversed(Path(opts["Paths"]["save_path"]).parents) if p.name == 'output'), None)
			init_net_path = output_directory / init_net
			if not init_net_path.exists():
				raise Exception(f"Initialization Network '{init_net}' is either not a correct path or not an existing run")
		init_net_opts = utils.read_config(Path(__file__).resolve().parent/'output'/init_net)
		init_net_setup = init_net_opts['Network']['Basic Setup']
		init_net_data = init_net_opts['Training']['Basic Setup']
		init_network_sparsity = init_net_opts['Network']['Optional']["Sparsity"]

		fit_variance_only = partial_bayes_params["fit_variance_only"]
		bayesian_rate = partial_bayes_params["bayesian_rate"]
		largest_grad = partial_bayes_params["largest_grad"]
		select_random_grads = partial_bayes_params["select_random_node"]
		bayes_params= opts["Training"]["Bayesian Parameters"]
		prior_mu = bayes_params["prior_mu"]
		prior_variance = bayes_params["prior_variance"]
		posterior_mu = bayes_params["posterior_mu"]
		posterior_rho = bayes_params["posterior_rho"]

		### LOAD THE INITIALIZATION NETWORK 
		network, *_ = model.get_model(model_type=init_net_setup["network_type"], net_architecture=init_net_setup["architecture"],
										layer_type=init_net_setup["layer_type"], task=init_net_setup["task"], 
										batch_norm=init_net_setup["use_batchnorm"], bias=init_net_setup["bias"], 
										input_size=init_net_setup["input_size"], output_size=init_net_setup["output_size"],
										kernel_size=init_net_setup["kernel_size"], stride=init_net_setup["stride"], 
										feature_size=init_net_setup["feature_size"], pooling_size=init_net_setup["pooling_size"], 
										padding_size=init_net_setup["padding_size"], dataset=init_net_data["dataset"],
										nonlinearity=init_net_setup["nonlinearity"], init_w=init_net_setup["init_weights"],						
										prior_mu=prior_mu, prior_variance=prior_variance, posterior_mu=posterior_mu, 
										posterior_rho=posterior_rho, sparse=init_network_sparsity, fit_variance_only=fit_variance_only)
			
		network, _ = utils.load_checkpoint(net=network, init_path=init_net, fold=fold, cont_run=True)
		
		if fold is not None:
			kf = KFold(n_splits=kfold, shuffle=False)
			train_index, _ = next(islice(kf.split(data_loader), fold, None), None)
			train_dataset = torch.utils.data.Subset(data_loader, train_index)
			train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			train_data_iter = tqdm(train_loader, position=2)
		else:
			train_data_iter = tqdm(data_loader, position=2)

		get_sensitivity = utils.SensitivityAnalyzer(network, bayesian_rate, largest_grad, select_random_grads, device)
		init_params, num_bayesian_params, total_num_params = get_sensitivity.compute_sensitivity(train_data_iter, criterion, task, dataset_setup, multi_label)
		return init_params, num_bayesian_params, total_num_params, partial_bayes_params
	
	fold = 0 if kfold > 1 else None 
	init_params, num_bayesian_params, total_num_params, partial_bayes_params = initialize_partial_bayes(fold)

if "bayesian" in network_type:
	kl_weight = opts["Training"]["Bayesian Parameters"]["kl_weight"]
	m_val_batch, len_val_data = None, None 
	if run_val and kfold == 1:
		m_val_batch=len(val_loader)
		len_val_data=len(val_loader.dataset)
	len_train_data = len(data_loader) if kfold > 1 else len(data_loader.dataset)
	kl_weight = utils.KL_Weight(m_train_batch=len(data_loader), 
								m_val_batch=m_val_batch, 
								len_train_data=len_train_data,
								len_val_data=len_val_data,
								method=kl_weight, 
								num_epochs=num_epochs,
								hi2low=opts["Training"]["Bayesian Parameters"]["kl_weight_hi2low"], 
								kl_weight_lim=opts["Training"]["Bayesian Parameters"]["kl_weight_lim"])

	num_samples = opts["Training"]["Bayesian Parameters"]["num_samples"]
	prior_mu = opts["Training"]["Bayesian Parameters"]["prior_mu"]
	prior_variance = opts["Training"]["Bayesian Parameters"]["prior_variance"]
	posterior_mu = opts["Training"]["Bayesian Parameters"]["posterior_mu"]
	posterior_rho = opts["Training"]["Bayesian Parameters"]["posterior_rho"]
	if "partial" in network_type:
		fit_variance_only = partial_bayes_params["fit_variance_only"]
	else: 
		fit_variance_only = False
else: 
	prior_mu, prior_variance, posterior_mu, posterior_rho, kl_weight, fit_variance_only = [None]*6

network, tmp["input_size"], tmp["feature_size"], tmp["output_size"] = model.get_model(model_type=network_type, net_architecture=net_architecture,
																					output_size=output_size, feature_size=feature_size, bias=bias, 
																					layer_type=layer_type, task=task, dataset=dataset_setup, input_size=input_size,
																					initialize_params=init_params, init_w=init_weights, batch_norm=batch_norm, 
																					nonlinearity=nonlinearity, kernel_size=kernel_size, stride=stride, 
																					pooling_size=pooling_size, padding_size=padding_size,
																					prior_mu=prior_mu, prior_variance=prior_variance,
																					posterior_mu=posterior_mu, posterior_rho=posterior_rho, 
																					sparse=sparsity, fit_variance_only=fit_variance_only)
if continue_run: 
	network, epoch_start = utils.load_checkpoint(network, init_network, cont_run=True) #continue_run=True to load the last checkpoint
else:
	epoch_start = 0
network.to(device)
"""======================================================================================="""
## OPTIMIZATION SETUP
if network_type=='partial_bayesian' and optimizer_type == 'sparseAdam':
	optimizer, optimizer_sparse, *parameters= utils.get_optimizer(network, opt_type=optimizer_type, lr=lr, sensitivity=init_params)
	_, param_sparse, _ = parameters 
	optimizers = optimizer, optimizer_sparse
else: 
	optimizer = utils.get_optimizer(network, opt_type=optimizer_type, lr=lr)
	optimizers = optimizer

lr_scheduler = utils.get_lr_scheduler(opts["Training"]["LR Scheduler"], optimizer)
if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau" and not run_val:
	raise Exception("If ReduceLROnPlateau is selected as a scheduler, run_val flag cannot be false")
"""======================================================================================="""
## LOGGING
if not os.path.isdir(opts["Paths"]['save_path']):
	os.makedirs(opts["Paths"]['save_path'])
	os.chdir(opts["Paths"]['save_path'])
	with open('config.json', 'w') as fp:
		json.dump(opts, fp)

metrics, logging_keys = utils.get_logging_keys(network_type, task, run_val, dataset=dataset_setup) 
full_log  = utils.CSVlogger(Path(opts["Paths"]['save_path'])/"log.csv", logging_keys)
"""======================================================================================="""
### START TRAINING
full_training_start_time = time.time()

def train_single_fold(network, train_loader, val_loader, optimizers, output_path, full_log, fold=0, network_type="regression"): 
	epoch_iter = trange(epoch_start, num_epochs, position=1)
	if isinstance(optimizers, tuple):
		optimizer = optimizers[0]
		optimizer_sparse = optimizers[1]
	else: 
		optimizer = optimizers
	for epoch in epoch_iter:
		network.train()
		epoch_time = time.time()
		train_data_iter = tqdm(train_loader, position=2)
		loss = 0
		total_loss = 0
		total_rmse = 0
		total_std = 0
		total_dice = 0
		total_iou = 0
		total_acc = 0
		total_kl = 0 
		for step, (inputs, targets, *args) in enumerate(train_data_iter): 
			inputs = inputs.to(device)
			targets = targets.to(device)

			one_hot_targets = False
			if task == 'classification' and not multi_label:
				targets = utils.get_one_hot(targets, num_classes=output_size, dataset=dataset_setup)
				one_hot_targets = True

			if network_type == 'deterministic':
				outputs = network(inputs, nll_loss=nll_loss)
				loss = criterion(outputs, targets)

				if task == "regression": 
					total_rmse += float(torch.sqrt(loss))

			elif 'bayesian' in network_type:  
				samples = []
				for i in range(num_samples):
					if i == 0:
						sample, kl = network.forward(inputs, nll_loss=nll_loss) 
					else: 
						sample = network.forward(inputs, return_kl=False, nll_loss=nll_loss)					
					samples.append(sample)
				
				outputs = torch.stack(samples, dim=0)
				total_kl += float(kl)
				
				if task == 'regression':
					rmse = []
					for i in range(outputs.shape[0]): 
						rmse.append(torch.sqrt(criterion(outputs[i], targets)))
					rmse = torch.stack(rmse, dim=0)
					likelihood = rmse.mean()
					total_rmse += float(likelihood)					
					std = rmse.std()
				else:
					likelihood = []
					for i in range(num_samples):
						likelihood.append(criterion(outputs[i], targets))
					likelihood = torch.mean(torch.stack(likelihood))
					std = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets, 
									 num_classes=output_size, task=task, multi_label=multi_label)
				
				total_std += float(std)
				if isinstance(optimizers, tuple):
					optimizer_sparse.zero_grad()

				loss = likelihood + kl_weight.get_weight(step, epoch)*kl
				
			if task == "classification": 
				#accuracy calculated as a mean of the batch
				total_acc += utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, multi_label)
			
			elif task == "segmentation": 
				if loss_func == 'ce' and dataset_setup != "cityscapes":
					dim = 1 if network_type == 'deterministic' else 2
					outputs = torch.nn.Softmax(dim=dim)(outputs)

				if dataset_setup == "cityscapes" or dataset_setup == "isic":
					total_iou += float(utils.calculate_iou(outputs, targets, num_classes=output_size))
				else:
					if loss_func == "dice":
						total_dice += float(1-loss)
					else:
						total_dice += float(utils.calculate_dice(input=outputs, target=targets, num_classes=output_size))

			total_loss += float(loss)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		metrics["Train Time"] = time.time()-epoch_time
		metrics['Train Loss'] = total_loss/(max(step, 1))
		metrics["Train Accuracy"] = total_acc / max(step, 1)
		metrics["Train RMSE"] = total_rmse / max(step, 1)
		metrics["Train Std"] = total_std / max(step, 1)
		metrics["Train KL"] = total_kl / max(step, 1)
		metrics["Train Brier Score"] = total_std / max(step, 1)
		metrics["Train Dice"] = total_dice / max(step, 1)
		metrics["Train IoU"] = total_iou / max(step, 1)
		output_metrics_train = [metrics[key] for key in logging_keys if 'Train' in key and "Best" not in key] 
		output_metrics_val = []
		metric_formats = {
			"classification": {
				"print_format": "Fold {kfold}, Epoch {epoch}: Train Loss={Train Loss:.4f}, KL={Train KL:.4f}, Accuracy={Train Accuracy:.4f}, Brier Score={Train Brier Score:.4e}",
			},
			"regression": {
				"print_format": "Fold {kfold}, Epoch {epoch}: Train Loss={Train Loss:.4f}, Train KL={Train KL:.4f}, Train RMSE={Train RMSE:.4f}, Train Std={Train Std:.4f}",
			},
			"segmentation": {
				"print_format": "Fold {kfold}, Epoch {epoch}: Train Loss={Train Loss:.4f}, KL={Train KL:.4f}, Dice={Train Dice:.4f}, IoU={Train IoU:.4f}",
			}
		}

		if not run_val and task == "segmentation":
			utils.save_pred(inputs, targets, outputs, epoch, output_path, dataset_setup, True, *args)

		print(metric_formats[task]["print_format"].format(**metrics, kfold=fold, epoch=epoch))

		# Evaluate the model on the Val set after each epoch
		if run_val:
			val_data_iter = tqdm(val_loader, position=2)
			total_loss = 0
			total_rmse = 0
			total_std = 0
			total_acc = 0 #accuracy
			total_dice = 0
			total_iou = 0
			total_kl = 0
			epoch_time = time.time()
			network.eval()
			with torch.no_grad():
				for step, (inputs, targets, *args) in enumerate(val_data_iter):
					inputs = inputs.to(device)
					targets = targets.to(device)

					one_hot_targets = False
					if task == "classification" and not multi_label:
						targets = utils.get_one_hot(targets, num_classes=output_size, dataset=dataset_setup)
						one_hot_targets = True

					if network_type == 'deterministic':
						outputs = network(inputs, nll_loss=nll_loss)
						loss = criterion(outputs, targets)

						if task == "regression": 
							total_rmse += float(torch.sqrt(loss))

					elif 'bayesian' in network_type:  
						samples = []
						for i in range(num_samples):
							if i == 0:
								sample, kl = network.forward(inputs, nll_loss=nll_loss) 
							else: 
								sample = network.forward(inputs, return_kl=False, nll_loss=nll_loss)					
							samples.append(sample)
						
						outputs = torch.stack(samples, dim=0)
						total_kl += float(kl)

						if task == 'regression':
							rmse = []
							for i in range(outputs.shape[0]): 
								rmse.append(torch.sqrt(criterion(outputs[i], targets)))
							rmse = torch.stack(rmse, dim=0)
							likelihood = rmse.mean()
							total_rmse += float(likelihood)					
							std = rmse.std()
						else:
							likelihood = []
							for i in range(num_samples):
								likelihood.append(criterion(outputs[i], targets))
							likelihood = torch.mean(torch.stack(likelihood))
							std = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets,
									   num_classes=output_size, task=task, multi_label=multi_label)
						
						total_std += float(std)
						if isinstance(optimizers, tuple):
							optimizer_sparse.zero_grad()

						loss = likelihood + kl_weight.get_weight(step, epoch)*kl
						
					if task == "classification": 
						#accuracy calculated as a mean of the batch
						total_acc += utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, multi_label)

					elif task == "segmentation": 
						if loss_func == 'ce' and dataset_setup != "cityscapes":
							dim = 1 if network_type == 'deterministic' else 2
							outputs = torch.nn.Softmax(dim=dim)(outputs)

						if dataset_setup == "cityscapes" or dataset_setup == "isic":
							total_iou += float(utils.calculate_iou(outputs, targets, output_size))
						else:
							if loss_func == "dice":
								total_dice += float(1-loss)
							else:
								total_dice += float(utils.calculate_dice(input=outputs, target=targets, num_classes=output_size))

					total_loss += float(loss)

		if run_val:
			
			metrics["Val Time"] = time.time()-epoch_time
			metrics['Val Loss'] = total_loss/max(step, 1)	
			metrics["Val Accuracy"] = total_acc / max(step, 1)
			metrics["Val RMSE"] = total_rmse / max(step, 1)
			metrics["Val Std"] = total_std / max(step, 1)
			metrics["Val KL"] = total_kl / max(step, 1)
			metrics["Val Brier Score"] = total_std / max(step, 1)
			metrics["Val Dice"] = total_dice / max(step, 1)
			metrics["Val IoU"] = total_iou / max(step, 1)

			metric_formats = {
				"classification": {
					"print_format": "Fold {kfold}, Epoch {epoch}: Val Loss={Val Loss:.4f}, KL={Val KL:.4f}, Accuracy={Val Accuracy:.4f}, Brier Score={Val Brier Score:.4e}"
				},
				"regression": {
					"print_format": "Fold {kfold}, Epoch {epoch}: Val Loss={Val Loss:.4f}, Val KL={Val KL:.4f}, Val RMSE={Val RMSE:.4f}, Val Std={Val Std:.4f}"
				},
				"segmentation": {
					"print_format": "Fold {kfold}, Epoch {epoch}: Val Loss={Val Loss:.4f}, KL={Val KL:.4f}, Dice={Val Dice:.4f}, IoU={Val IoU:.4f}"
				}
			}

			print(metric_formats[task]["print_format"].format(**metrics, kfold=fold, epoch=epoch))

			if task == "segmentation":
				utils.save_pred(inputs, targets, outputs, epoch, output_path, dataset_setup, False, *args)

			output_metrics_val = [metrics[key] for key in logging_keys if 'Val' in key and "Best" not in key] 

		full_log.write([fold, epoch] + output_metrics_train + output_metrics_val)
		
		save_dict = {'epoch': epoch+1, 'network_state_dict':network.state_dict(), 'current_train_time': time.time()-full_training_start_time,
				'optim_state_dict':optimizer.state_dict()}
			
		best_metrics = {
			"classification": {"train": ("Train Accuracy", "Best Train Accuracy", operator.gt), "val": ("Val Accuracy", "Best Val Accuracy", operator.gt)},
			"regression": {"train": ("Train RMSE", "Best Train RMSE", operator.lt), "val": ("Val RMSE", "Best Val RMSE", operator.lt)},
			"segmentation": {"train": ("Train Dice", "Best Train Dice", operator.gt), "val": ("Val Dice", "Best Val Dice", operator.gt)},
			"cityscapes": {"train": ("Train IoU", "Best Train IoU", operator.gt), "val": ("Val IoU", "Best Val IoU", operator.gt)},
			"isic": {"train": ("Train IoU", "Best Train IoU", operator.gt), "val": ("Val IoU", "Best Val IoU", operator.gt)}
		}

		if dataset_setup == "cityscapes" or dataset_setup == 'isic':
			metric_key, best_metric_key, comparison_op = best_metrics[dataset_setup]["val"] if run_val else best_metrics[dataset_setup]["train"]
		else: 
			metric_key, best_metric_key, comparison_op = best_metrics[task]["val"] if run_val else best_metrics[task]["train"]

		if comparison_op(metrics[metric_key], metrics[best_metric_key]):
			metrics[best_metric_key] = metrics[metric_key] 
			if run_val:
				if kfold > 1:
					torch.save(save_dict, output_path / f'checkpoint_best_val_kfold{fold}.pth.tar')
				else:
					torch.save(save_dict, output_path / 'checkpoint_best_val.pth.tar')
			else:
				torch.save(save_dict, output_path / 'checkpoint_best_train.pth.tar')

		if kfold > 1:
			torch.save(save_dict, output_path / f'checkpoint_kfold{fold}.pth.tar')
		else:
			torch.save(save_dict, output_path / 'checkpoint.pth.tar')

		if lr_scheduler is not None:
			if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
				if run_val:
					lr_scheduler.step(metrics['Val Loss'])
				else:
					lr_scheduler.step(metrics['Train Loss'])
			else:
				lr_scheduler.step()

			print(f"Learnign rate: {lr_scheduler.get_last_lr()}")

if kfold > 1: 
	print(f"Running {kfold} kfold Cross-validation")
	kf = KFold(n_splits=kfold, shuffle=False)
	for fold, (train_index, val_index) in enumerate(kf.split(data_loader)):
		print(f"Running fold {fold+1}")
		# Split the data into train and Val sets
		train_dataset = torch.utils.data.Subset(data_loader, train_index)
		val_dataset = torch.utils.data.Subset(data_loader, val_index)

	   # Create a DataLoaders
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

		if fold > 0 :
			if network_type == "partial_bayesian":
				init_params, *_ = initialize_partial_bayes(fold=fold)
			network, tmp["input_size"], tmp["feature_size"], tmp["output_size"] = model.get_model(model_type=network_type, task=task, net_architecture=net_architecture,
																							layer_type=layer_type, initialize_params=init_params, dataset=dataset_setup, 
																							padding_size=padding_size, init_w=init_weights, batch_norm=batch_norm, 
																							kernel_size=kernel_size, stride=stride, pooling_size=pooling_size, 
																							input_size=input_size, feature_size=feature_size, output_size=output_size,
																							bias=bias, nonlinearity=nonlinearity, prior_mu=prior_mu, 
																							prior_variance=prior_variance, posterior_mu=posterior_mu, posterior_rho=posterior_rho, 
																							sparse=sparsity)
			
			network.to(device)
			optimizer = utils.get_optimizer(network, opt_type=optimizer_type, lr=lr, weight_decay=weight_decay)
		train_single_fold(network, train_loader, val_loader, optimizers=optimizer, full_log=full_log,
						  output_path=Path(opts["Paths"]['save_path']), fold=fold, network_type=network_type)
		
else: 
	print("Running a single split training loop")
	train_single_fold(network, data_loader, val_loader, optimizers=optimizers, full_log=full_log,
					output_path=Path(opts["Paths"]['save_path']), fold=0, network_type=network_type)
	
if not opt.no_date:
	log_run = utils.log_runs(opts)