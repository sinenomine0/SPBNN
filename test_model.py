import argparse, random, time, os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd 
import dataset
import models.model as model
import utils
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.utils.data
"""======================================================================================="""
parse_in = argparse.ArgumentParser()
parse_in.add_argument('--run',      type=str, default=None, help='which model to run testing on')
parse_in.add_argument('--dataset',  type=str, default=None, help='which dataset to run testing on')
parse_in.add_argument('--noise',    type=tuple, default=None, help='add noise to the dataset to check for robustness')
parse_in.add_argument('--save_seg', action='store_true', default=False, help='save tensor predictions for segmentation task')

opt = parse_in.parse_args()
save_seg = True if opt.save_seg else False
run = opt.run
run_folder = Path(__file__).resolve().parent / 'output'/ run
opts = utils.read_config(run_folder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opts["Paths"]["dataset_path"] != '':
	dataset_path = opts["Paths"]["dataset_path"]
else:
	dataset_path = os.path.join(Path(__file__).resolve().parent, 'datasets')

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
cont_interrupt = tmp["continue_run"]
dataset_setup = tmp['dataset'] if opt.dataset is None else opt.dataset
noise = opt.noise
num_points = tmp["num_points"]

### TRAINING PARAMETERS
tmp = opts["Training"]["Basic Parameters"]
optimizer_type = tmp["optimizer"]
lr = tmp["lr"]
kfold = tmp["kfold"]
cuda_id = tmp["gpu"]
seed = tmp["seed"]
run_val = tmp["run_val"]
train_val_split = tmp["train_val_split"]
num_epochs = tmp["num_epochs"]
weight_decay = tmp["weight_decay"]
loss_func = tmp["loss_func"]
nll_loss = True if "nll" in loss_func else False #needed to check if log_softmax is needed before loss calculation
multi_label = True if loss_func == "bceLogits" and output_size > 2 else False
criterion = utils.get_loss_func(loss_func) 
init_params = None 
tmp = opts["Network"]["Basic Setup"]
sparsity = opts["Network"]["Optional"]['Sparsity']


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')

def turn_off_gradients(model):
	for param in model.parameters():
		param.requires_grad_(False)

flatten_input = True if layer_type == 'linear' and task == "classification" else False
data_loader = dataset.get_dataloader(root=dataset_path, dataset=dataset_setup, kfold=kfold,
									batch_size=1, train_shuffle=True, 
									num_points=num_points, input_size=input_size,
									train_split=train_val_split, 
									run_val=run_val, test=True, noise=noise,
									flatten_input=flatten_input)

if "bayesian" in network_type:
	num_samples = opts["Training"]["Bayesian Parameters"]["num_samples"]
	prior_mu = opts["Training"]["Bayesian Parameters"]["prior_mu"]
	prior_variance = opts["Training"]["Bayesian Parameters"]["prior_variance"]
	posterior_mu = opts["Training"]["Bayesian Parameters"]["posterior_mu"]
	posterior_rho = opts["Training"]["Bayesian Parameters"]["posterior_rho"]
else: 
	prior_mu, prior_variance, posterior_mu, posterior_rho, kl_weight = [None]*5

test=True if network_type == 'partial_bayesian' else False
network, tmp["input_size"], tmp["feature_size"], tmp["output_size"] = model.get_model(model_type=network_type, net_architecture=net_architecture, 
																					  dataset=dataset_setup, batch_norm=batch_norm, layer_type=layer_type,
																					  kernel_size=kernel_size, stride=stride, pooling_size=pooling_size, 
																					  padding_size=padding_size, feature_size=feature_size, input_size=input_size,
																					  output_size=output_size, initialize_params=init_params, init_w=init_weights, 
																					  nonlinearity=nonlinearity, sparse=sparsity, task=task,
																					  bias=bias, prior_mu=prior_mu, prior_variance=prior_variance, 
																					  posterior_mu=posterior_mu, posterior_rho=posterior_rho, test=test)
network.to(device)
metrics, logging_keys = utils.get_logging_keys(network_type, task, run_val, test=True, dataset=dataset_setup) 
log_file_name = "test" if opt.dataset is None else f"test_{opt.dataset}"
log_file_name = log_file_name + f"Noise_{noise[0]}_{noise[-1]}.csv" if noise else log_file_name + '.csv'
full_log  = utils.CSVlogger(run_folder/log_file_name, logging_keys)
"""======================================================================================="""
### START TESTING
def test_single_fold(network, test_loader, output_path, fold=0):
	test_data_iter = tqdm(test_loader, position=2)
	network.eval()
	with torch.no_grad():
		turn_off_gradients(network)
		confidences_, true_classes, predictions = [], [], []
		for step, (inputs, targets, *args) in enumerate(test_data_iter):
			inputs = inputs.to(device)
			targets = targets.to(device)
			one_hot_targets = False
			
			if task != 'segmentation' and not multi_label:
				metrics['Y'] = int(targets)
			if task == "classification" and not multi_label:
				targets = utils.get_one_hot(targets, num_classes=output_size, dataset=dataset_setup)
				one_hot_targets = True
		
			inference_time = time.time()
			
			if network_type == 'deterministic':
				outputs = network(inputs, nll_loss=nll_loss)
				metrics['Test Time'] = time.time() - inference_time
				loss = criterion(outputs, targets)

				if task == 'regression':
					metrics['Y_pred'] = float(outputs)
					metrics['Test RMSE'] = float(torch.sqrt(loss))
				elif task == 'classification': 
					predicted = torch.argmax(outputs, -1)
					if nll_loss:
						#invert the log_ in log_softmax
						conf = torch.exp(outputs)
					else: 
						if multi_label:
							conf = F.sigmoid(outputs)
						else:
							conf = F.softmax(outputs, dim=-1) 

			elif 'bayesian' in network_type:  
				samples = []
				for i in range(num_samples):
					if i == 0:
						sample, kl = network.forward(inputs, nll_loss=nll_loss) 
					else: 
						sample = network.forward(inputs, return_kl=False, nll_loss=nll_loss)					
					samples.append(sample)
				metrics['Test Time'] = time.time() - inference_time
				
				outputs = torch.stack(samples, dim=0)
				metrics['Test KL'] = float(kl)
				if task == 'regression':
					rmse = []
					for i in range(outputs.shape[0]): 
						rmse.append(torch.sqrt(criterion(outputs[i], targets)))
					rmse = torch.stack(rmse, dim=0)
					likelihood = rmse.mean()			
					std = rmse.std()
				else:
					likelihood = []
					for i in range(num_samples):
						likelihood.append(criterion(outputs[i], targets))
					likelihood = torch.mean(torch.stack(likelihood))
					std = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets, 
												num_classes=output_size, task=task, multi_label=multi_label)
				
				loss = likelihood + kl
				std = float(std)
				
				if task == 'regression':
					metrics['Y_pred'] = float(outputs.mean())
					metrics['Test RMSE'] = float(likelihood.mean())
					metrics['Test Std'] = std
					
				elif task == 'classification': 
					if nll_loss:
						#invert the log_ in log_softmax
						conf = torch.exp(outputs)
					else: 
						if multi_label:
							conf = F.sigmoid(outputs)
						else:
							conf = F.softmax(outputs, dim=-1) 
					predicted = torch.mode(torch.argmax(conf, -1), dim=0).values

			metrics['Test Loss'] = float(loss)
			if task == "classification":            
				metrics['Test Accuracy'] = utils.calculate_accuracy(outputs, targets, nll_loss, one_hot_targets, multi_label)
				metrics['Y_pred'] = int(predicted)
				if multi_label: 
						metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"] = utils.calculate_tp_tn_fp_fn(outputs, targets)
						true_classes.append(targets)
						predictions.append(conf)
			
			elif task == "segmentation": 
				if loss_func == 'ce' and dataset_setup != "cityscapes":
					dim = 1 if network_type == 'deterministic' else 2
					outputs = torch.nn.Softmax(dim=dim)(outputs)
					conf = outputs.clone()
				else:
					true_class_indices = torch.argmax(outputs, dim=1)

			if "bayesian" in network_type and task != "regression": 
				metrics['Test Brier Score'] = std
				metrics['Y_conf'] = float(conf.mean(dim=(0)).max()) ## confidence of the true class
			else: 
				if task == 'classification':
					metrics['Y_conf'] = float(conf.max()) ## confidence of the true class
				if task != 'regression':
					metrics['Test Brier Score'] = utils.compute_brier_score(outputs, targets, nll_loss, one_hot_targets, 
							num_classes=output_size, task=task, multi_label=multi_label)
			if task == 'segmentation':
				if dataset_setup != 'cityscapes' and dataset_setup != 'isic':
					metrics['Test Dice'] = float(utils.calculate_dice(input=outputs, target=targets, num_classes=output_size))
				else:
					metrics['Test IoU'] = float(utils.calculate_iou(input=outputs, target=targets, num_classes=output_size))
			metrics['Test Entropy'] = float(utils.predictive_entropy(outputs, nll_loss=nll_loss, task=task, multi_label=multi_label)[0])
					
			full_log.write([metrics[key] for key in logging_keys])

			if task == 'segmentation':
				pred_path = run_folder / f'test_preds_{fold}'
				if not pred_path.exists():
					pred_path.mkdir()
				if step % 100 == 0:
					utils.save_pred(inputs, targets, outputs, step, pred_path, dataset_setup, False, *args)
				if save_seg:
					utils.append_to_hdf5(f"predictions_{fold}.h5", pred_path, step, inputs, targets, outputs, *args)
	if multi_label:
		with open(run_folder/f'test_predictions_{fold}.pt', 'ab') as f:
			torch.save(torch.stack(predictions), f)
		with open(run_folder/f'test_targets_{fold}.pt', 'ab') as f:
			torch.save(torch.stack(true_classes), f)

if kfold > 1: 
	print(f"Running test for {kfold} folds")
	kf = KFold(n_splits=kfold, shuffle=False)
	for fold, (train_index, test_index) in enumerate(kf.split(data_loader)):
		network, *_ = model.get_model(model_type=network_type, net_architecture=net_architecture, 
								dataset=dataset_setup, batch_norm=batch_norm, layer_type=layer_type,
								kernel_size=kernel_size, stride=stride, pooling_size=pooling_size, 
								padding_size=padding_size, feature_size=feature_size, input_size=input_size,
								output_size=output_size, initialize_params=init_params, init_w=init_weights, 
								nonlinearity=nonlinearity, sparse=sparsity, task=task,
								bias=bias, prior_mu=prior_mu, prior_variance=prior_variance, 
								posterior_mu=posterior_mu, posterior_rho=posterior_rho, test=test)
		test_dataset = torch.utils.data.Subset(data_loader, test_index)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
		net_fold, _ = utils.load_checkpoint(network, run_folder, cont_run=True, fold=fold)
		net_fold.to(device)
		print(f"Running test for fold {fold}")
		test_single_fold(net_fold, test_loader, run_folder, fold=fold)
else:
	net_fold, _ = utils.load_checkpoint(network, run_folder, cont_run=True)
	test_single_fold(net_fold, data_loader, run_folder)

df = pd.read_csv(f"{str(run_folder)}/{log_file_name}")
means = df.mean() 
if task == 'classification':
	headings = 'Test Time\tTest Accuracy\tTest Entropy'
	values = f'{means["Test Time"]:0.4f}\t\t{means["Test Accuracy"]:.4f}\t\t{means["Test Entropy"]:.4f}'
elif task == 'segmentation':
	if dataset_setup != 'cityscapes' and dataset_setup != 'isic':
		headings = 'Test Time\tTest Dice\tTest Entropy'
		values = f'{means["Test Time"]:0.4f}\t\t{means["Test Dice"]:.4f}\t\t{means["Test Entropy"]:.4f}'
	else:
		headings = 'Test Time\tTest IoU\tTest Entropy'
		values = f'{means["Test Time"]:0.4f}\t\t{means["Test IoU"]:.4f}\t\t{means["Test Entropy"]:.4f}'
elif task == 'regression':
	headings = 'Test Time\tTest RMSE'
	values = f'{means["Test Time"]:0.4f}\t\t{means["Test RMSE"]:.4f}'

if "bayesian" in network_type:
	if task!="regression": 
		headings = headings + '\tTestBrierScore'
		values = values + f'\t\t{means["Test Brier Score"]:.4f}'
	else: 
		headings = headings + '\tTestStd'
		values = values + f'\t\t{means["Test Std"]:.4f}'
print(headings)
print(values)
print(f"Completed running test on run {run}")