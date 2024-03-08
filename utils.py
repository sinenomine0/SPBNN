import csv, os, json, glob, math, h5py
from pathlib import Path
import torch
import numpy as np
from collections.abc import Iterable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import helpers.isic_dataset as isic

"""==================================================="""
"""================ Metrics Utilities ================"""
"""==================================================="""
def calculate_tp_tn_fp_fn(input, target):
	"""
	Compute True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
	for multi-label classification.

	Parameters:
	- target (torch.Tensor): Ground truth labels, shape (n_samples, n_classes)
	- input (torch.Tensor): Predicted labels, shape (n_samples, n_classes)

	Returns:
	- tp (torch.Tensor): Total True Positives
	- tn (torch.Tensor): Total True Negatives
	- fp (torch.Tensor): Total False Positives
	- fn (torch.Tensor): Total False Negatives
	"""
	probs = F.sigmoid(input)
	# Ensure both tensors have the same shape
	if not target.shape == input.shape:
		input = input.mean(dim=0)

	preds = (probs > 0.5).float()

	# Compute True Positives (TP)
	tp = (target * preds).sum()

	# Compute True Negatives (TN)
	tn = ((1 - target) * (1 - preds)).sum()

	# Compute False Positives (FP)
	fp = ((1 - target) * preds).sum()

	# Compute False Negatives (FN)
	fn = (target * (1 - preds)).sum()

	# # Normalize the results to ensure everything is between 0 and 1
	# tp = tp / target.sum() if tp!=0 else tp
	# tn = tn / (target.numel() - target.sum()) if tn!=0 else tn
	# fp = fp / (preds-target > 0).sum() if fp!=0 else fp 
	# fn = 

	return float(tp), float(tn), float(fp), float(fn)

def get_one_hot(targets: torch.Tensor, num_classes: int, dataset: str=""):
	"""
	Converts an integer label torch.autograd.Variable to a one-hot Variable.

	Args:
		targets (torch.autograd.Variable): A tensor of shape (N, 1) containing labels.
		num_classes (int): The number of classes in the labels.

	Returns:
		torch.FloatTensor: A tensor of shape (N, num_classes) with labels in one-hot format.
	"""
	one_hot = F.one_hot(targets.to(torch.long), num_classes=num_classes).float()
	if dataset == 'cityscapes':
		return one_hot.permute(0, 3, 1, 2)
	return one_hot

def calculate_iou(input: torch.Tensor, target: torch.Tensor, num_classes: int):
	"""
	Calculate Intersection over Union (IoU) for multi-class classification using PyTorch.

	Parameters:
		target (torch.Tensor): Ground truth labels (2D tensor).
		input (torch.Tensor): Predicted softmax output (2D tensor).
		num_classes (int): Number of classes.

	Returns:
		float: Mean IoU across all classes.
	"""
	iou_scores = []

	dim = 2 if len(input.shape) > 4 else 1
	pred = torch.argmax(input, dim=dim)
	if target.shape[1] > 1: 
		target = torch.argmax(target.clone(), dim=1)
	
	if dim == 2: 
		target = target.expand_as(pred)

	for class_id in range(1, num_classes):
		#ignore class index 0 as it is associated with void classes
		true_class = (target == class_id).float()
		pred_class = (pred == class_id).float()

		intersection = torch.sum(torch.logical_and(true_class, pred_class)).item()
		union = torch.sum(torch.logical_or(true_class, pred_class)).item()

		if union == 0: #if ground truth is empty
			iou_scores.append(1.0 if torch.sum(pred_class).item() == 0 else 0.0)
		else:
			iou_scores.append(intersection / union)

	mean_iou = torch.mean(torch.tensor(iou_scores))
	return mean_iou


def calculate_dice(input: torch.Tensor, target: torch.Tensor, num_classes: int, epsilon=1e-5):
	"""
	Compute the Dice coefficient for multi-class segmentation.

	Args:
		input (torch.Tensor): Predicted tensor with shape (batch_size, num_classes, height, width).
		target (torch.Tensor): Ground truth tensor with shape (batch_size, num_classes, height, width).
		num_classes (int): Number of classes.
		epsilon (float): Smoothing term to avoid division by zero.

	Returns:
		float: Mean Dice coefficient value across all classes.
	"""
	targets_ = target.expand_as(input)
	intersection = input * targets_ 
	union = input + targets_
	dice = (2.0 * intersection + epsilon) / (union + epsilon)
	return dice.mean().item()

def compute_brier_score(logits: torch.Tensor, targets: torch.Tensor, 
						nll_loss: bool=False, one_hot_targets: bool=False,
						num_classes: int=2, task: str='classification', 
						multi_label: bool=False):
	"""
	Compute the Brier score.

	Args:
	- logits (torch.Tensor): Raw scores from the model.
	- targets (torch.Tensor): Ground truth labels.
	- nll_loss (bool): whether log_softmax layer is already applied to logits

	Returns:
	- float: Brier score.
	"""
	if multi_label: 
		probs = F.sigmoid(logits)
		targets_ = targets
	else:
		if task == 'classification' or task == 'segmentation':
			# Apply softmax to obtain probabilities
			if nll_loss:
				probs = torch.exp(logits)
			else:
				dim = -1 if task == 'classification' else -3
				probs = F.softmax(logits, dim=dim)
				
		# One-hot encode the targets
		if not one_hot_targets and task == 'classification': 
			targets_ = F.one_hot(targets, num_classes=num_classes)
		else: 
			targets_ = targets
	targets_ = targets_.expand_as(probs)
	# Compute Brier score
	class_dim = -1 if task == 'classification' else -3
	brier_score = torch.sum((probs - targets_.float())**2, dim=class_dim) / num_classes

	return torch.mean(brier_score).item()

def expected_calibration_error(target, y_prob, num_bins=10):
	"""
	Compute the Expected Calibration Error (ECE) for a classification model.

	Parameters:
	target (array-like): True labels (0 or 1).
	y_prob (array-like): input probabilities for the positive class.
	num_bins (int): Number of bins to divide the data into for calibration.

	Returns:
	ece (float): The Expected Calibration Error.
	"""
	# Ensure target and y_prob have the same length
	if len(target) != len(y_prob):
		raise ValueError("Input arrays must have the same length.")

	# Initialize variables to store ECE and total number of samples
	ece = 0.0
	total_samples = len(target)

	# Calculate bin edges
	bin_edges = np.linspace(0, 1, num_bins + 1)

	# Calculate bin widths
	bin_widths = np.diff(bin_edges)

	for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:]):
		# Filter samples falling within the current bin
		in_bin = (y_prob >= bin_start) & (y_prob < bin_end)

		# Number of samples in the current bin
		num_samples_in_bin = np.sum(in_bin)

		if num_samples_in_bin > 0:
			# Calculate the average input probability in the bin
			avg_input_prob = np.mean(y_prob[in_bin])

			# Calculate the average true probability in the bin
			avg_true_prob = np.mean(target[in_bin])

			# Calculate the absolute difference between input and true probabilities
			ece += (num_samples_in_bin / total_samples) * np.abs(avg_input_prob - avg_true_prob)

	return ece


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor,
						nll_loss: bool=False, one_hot_targets: bool=False, 
						multi_label: bool=False):
	if multi_label:
		probs = F.sigmoid(output)
		pred_binary = probs > 0.5
		if pred_binary.shape != target.shape:
			pred_binary = torch.mode(pred_binary, dim=0).values
		accuracy = (pred_binary == target).sum()/target.shape[-1]
	else: 
		if nll_loss: 
			probs = torch.exp(output, dim=-1)
		else:
			probs = F.softmax(output, dim=-1)

		if one_hot_targets:
			target_ = torch.argmax(target, dim=1)
		else: 
			target_ = target
		pred = torch.argmax(probs, -1)
		
		if len(pred.shape) > 1:
			#for Bayesian outputs
			# Calculate mode (most common) input class
			pred = torch.mode(pred, dim=0).values
		accuracy = (pred == target_).sum()
	return float(accuracy/target.shape[0])

def entropy_of_expected(probs, epsilon=1e-10, class_dim=-1):
	"""
	param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
	dim: class dimension
	return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
	"""
	mean_probs = np.mean(probs, axis=0)
	log_probs = -np.log(mean_probs + epsilon)
	return np.sum(mean_probs * log_probs, axis=class_dim)

def expected_entropy(probs, epsilon=1e-10, class_dim=-1):
	"""
	:param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
	:return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
	"""
	log_probs = -np.log(probs + epsilon)
	return np.mean(np.sum(probs * log_probs, axis=class_dim), axis=0)

def predictive_entropy(logits, nll_loss=False, task='', multi_label=False):
	if task != 'segmentation':
		if nll_loss: 
			probs = torch.exp(logits, dim=-1)
		else:
			if multi_label: 
				probs = F.sigmoid(logits)
			else:
				probs = F.softmax(logits, dim=-1)
	else:
		probs = logits
	dim = -3 if task == 'segmentation' else -1
	eoe = entropy_of_expected(probs.detach().cpu().numpy(), class_dim=dim)
	ee = expected_entropy(probs.detach().cpu().numpy(), class_dim=dim)
	return eoe.mean(), ee.mean()

"""==================================================="""
"""================ Sparse Utilities ================"""
"""==================================================="""
def get_sparse_mask(tensor_size, rate: float, kernel: int=0) -> torch.Tensor:
	"""
	Generates a mask of size (tensor_size) with (rate)*numel(tensor) elements = 1
	such that the elements with entry 1 will be in turn set as zero to generate
	a tensor with sparsity level (rate)

	Args:
		tensor_size (_type_): torch.Size or torch.Tensor.shape
		rate (float): sparsity rate (float value between 0 and 1)
					  if you want to report it as density rate - use (1-rate) as value
	Returns:
		_torch.Tensor_: tensor of size tensor_size, with (1-rate) number of elements nonzero
	"""
	assert 0 <= rate <= 1
	if len(tensor_size) == 1:
		x = tensor_size
		y = 1
	else:
		x, y = tensor_size
	if kernel == 0:		
		num_elements = int(x * y * rate)
		mask = torch.zeros((x, y), dtype=torch.bool)
		indices = torch.randperm(x * y)[:num_elements]
	else:
		x, y = tensor_size
		num_elements = int(x * y * kernel**2 * rate)
		mask = torch.zeros((x, y, kernel, kernel), dtype=torch.bool)
		indices = torch.randperm(x * y * kernel**2)[:num_elements]
	mask.view(-1)[indices] = True
	return mask
		
"""==================================================="""
"""================ Bayesian Utilities ================"""
"""==================================================="""
class KL_Weight():
	"""_summary_
	 REF: Weight Uncertainty in Neural Networks by Blundell (2015)
	 the weight of the KL term is set to 2^(M-i) / 2^M - 1
	 where i is the minibatch index, 
	 M is the total number of minibatches (length of the dataloader)
	"""
	def __init__(self, method: str, 
		  			m_train_batch: int, 
					m_val_batch: int,
					len_train_data: int,
					len_val_data: int,
					num_epochs: int,
					kl_weight_lim: list,
					hi2low=False):
		super().__init__()

		self.method = method
		self.init_weight = kl_weight_lim[0]
		self.targetweight = kl_weight_lim[1]
		self.num_epochs = num_epochs
		#controls whether to increase or decrease the weight a function of epochs
		# if hi2low (high to low) is set to true, the KL weight would be highest 
		# initially and decreasing with number of epochs -- implemented for method="epoch"
		self.h2l = hi2low 

		self.len_train_data = len_train_data
		self.len_val_data = len_val_data

		self.train_batch = m_train_batch
		self.val_batch = m_val_batch
	
	def get_weight(self, step:int, epoch: int, training=True):
		if self.method == "blundell": 
			if training:
				return 2**(self.train_batch-step) / (2**self.train_batch -1)
			else:
				return 2**(self.val_batch-step) / (2**self.val_batch -1)
			
		elif self.method == "num_samples": 
			if training: 
				return 1/self.len_train_data
			else: 
				return 1/self.len_val_data
			
		elif self.method == "epoch": 
			if self.h2l:
				return self.targetweight - (self.targetweight - self.init_weight) * (epoch / self.num_epochs)
			else:
				return self.init_weight + (self.targetweight - self.init_weight) * (epoch / self.num_epochs)
			
		elif self.method == "batch_size":
			return 1/self.train_batch
		
		elif isinstance(self.method, (float, int)):
			return self.method


class SensitivityAnalyzer:
	def __init__(self, model: nn.Module, rate: int, largest_grad=True, select_random=False, device='cuda', layer_wise=False):
		self.model = model
		self.device = device
		self.largest_grad = largest_grad
		self.bayes_rate = rate
		self.select_random = select_random
		self.layer_wise = True if layer_wise or rate == 0 else False
		self.total_num_params = 0
		self.total_num_bayes = 0 
		self.ref_value = 0
		self.sens_tensor = []
		self.out = {'names': [], 'params': [], 'grads': [], 'sensitivity': [], 'masks': []}

	def compute_sensitivity(self, data_iter, criterion, task: str='', dataset: str='', multi_label: bool=False):
		self.model.to(self.device)

		# Compute the partial gradient
		self.model.zero_grad()
		for _, (inputs, targets, *args) in enumerate(data_iter):
			inputs = inputs.to(self.device)
			targets = targets.to(self.device)
			if not task == 'regression' and dataset == 'cityscapes' and not multi_label:
				targets = get_one_hot(targets, num_classes=10, dataset=dataset)
			# Forward pass
			outputs = self.model(inputs)

			loss = criterion(outputs, targets)
			loss.backward(retain_graph=True)
			if self.layer_wise:
				break

		#GET PARAMETERS
		if hasattr(self.model, 'layers'):
			for i, layer in enumerate(self.model.layers): 
				name = layer.__class__.__name__
				if "ReLU" in name or "Pool" in name or "Instance": 
					continue

				if hasattr(layer, 'weight'):
					self.save_param_info(layer.weight, f"layer.{i}.{name}.weight")
					if layer.bias is not None:
						self.save_param_info(layer.bias, f"layer.{i}.{name}.bias")
				else:
					self.extract_block_params(layer, f"layer.{i}."+name)
		else:
			self.extract_block_params(self.model.encoder, "encoder")
			self.extract_block_params(self.model.decoder, "decoder")
			self.extract_block_params(self.model.classifier, "classifier")

		if not self.layer_wise:
			self.ref_value = self.compute_ref_value()

		for i, s in enumerate(self.out['sensitivity']):
			if self.layer_wise:
				layer='det'
				if i == len(self.out['sensitivity'])-4 or i == len(self.out['sensitivity'])-3:
					layer='bayes'
				self.out['masks'].append(self.get_mask(s, layer=layer))
			else:
				self.out['masks'].append(self.get_mask(s))
				
		return self.out, self.total_num_bayes, self.total_num_params

	def save_param_info(self, param: torch.Tensor, param_name:str):
		self.out["names"].append(param_name)
		self.out["grads"].append(param.grad)
		self.out["sensitivity"].append(torch.square(param.grad))
		if not "Norm" in param_name:
			if param.is_sparse:
				self.sens_tensor += torch.square(param.grad).coalesce().values().tolist()
			else:
				self.sens_tensor += torch.square(param.grad).flatten().tolist()  
		param.requires_grad = False
		self.out["params"].append(param)

	def extract_block_params(self, block, block_name):
		if not hasattr(block, "layers") and ("skip" not in block_name and "upconv" not in block_name and "classifier" not in block_name):
			if isinstance(block, Iterable):
				for i in range(len(block)):
					self.extract_block_params(getattr(block, str(i)), block_name+f'.{i}')
			else:
				children = ["upconv_layer", "convblock", "skip"]
				for child in children: 
					if hasattr(block, child):
						self.extract_block_params(getattr(block, child), block_name+f'.{child}')
				
		else:
			if "upconv" in block_name or "classifier" in block_name:
				self.save_param_info(block.weight, f"{block_name}.weight")
				if block.bias is not None:
					self.save_param_info(block.bias, f"{block_name}.bias")
			else:
				iterate_ = block.layers if "skip" not in block_name else block
				for i, layer in enumerate(iterate_):
					if hasattr(layer, 'weight'):
						if layer.weight is None: 
							continue
						name = block_name + f".{i}." + layer.__class__.__name__
						self.save_param_info(layer.weight, f"{name}.layer.{i}.weight")
						if layer.bias is not None:
							self.save_param_info(layer.bias, f"{name}.layer.{i}.bias")

	def get_mask(self, param_sens, layer=None):
		if self.layer_wise:
			if layer == 'bayes': 
				mask = torch.ones_like(param_sens).to(torch.bool)
			else:
				mask = torch.zeros_like(param_sens).to(torch.bool)
		else:
			if self.select_random:
				mask = get_sparse_mask(param_sens.size(), self.bayes_rate)
			else:
				param_sens = param_sens.to_dense() if param_sens.is_sparse else param_sens
				if self.largest_grad:
					mask = torch.ge(param_sens, self.ref_value)
				else: 
					mask = torch.le(param_sens, self.ref_value)
		self.total_num_bayes += int(mask.sum())
		self.total_num_params += int(mask.numel())
		## invert the mask since the mask is used for sparsifying
		## where the mask ==1, and we want the values that are outside 
		## the selected range to be set to 0 
		return ~mask

	def compute_ref_value(self):
		self.sens_tensor = torch.tensor(self.sens_tensor)
		k = int(self.bayes_rate * len(self.sens_tensor))
		values, _ = torch.topk(self.sens_tensor, k=k, largest=self.largest_grad)

		if self.largest_grad:
			ref_value = values.min()
			if ref_value == 0:
				# Find smallest nonzero value
				nonzero_values = values[torch.nonzero(values)]
				ref_value = nonzero_values.min()
		else:
			ref_value = values.max()
		return ref_value			

"""==================================================="""
"""================ Network Utilities ================"""
"""==================================================="""
class DiceLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		return 1-calculate_dice(input, target)

def get_nonlinearity(nonlinearity:str):
	if nonlinearity == "sigmoid":
		return nn.Sigmoid()
	elif nonlinearity == "relu":
		return nn.ReLU(inplace=True)
	elif nonlinearity == "softplus":
		return nn.Softplus()
	elif nonlinearity == 'tanh':
		return nn.Tanh()
	elif nonlinearity == 'leaky_relu':
		return nn.LeakyReLU()


def load_checkpoint(net:nn.Module, init_path: str, cont_run = False, fold=None):
	try:
		print(f"Network initializing with parameters from {init_path}")
		init_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'output', init_path))
	except:
		raise Exception(f'Error when loading initialization weights! Please make sure that weights exist at {init_path}!')

	if cont_run:
		if fold is not None:
			checkpoint = torch.load((init_path+f'/checkpoint_kfold{fold}.pth.tar'))
		else:
			checkpoint = torch.load((init_path+'/checkpoint.pth.tar'))
	else:
		os.chdir(init_path)
		list_checkpoints = glob.glob('checkpoint_best_*') 
		if len(list_checkpoints) == 0:
			print("No best performance checkpoint found. Will load any checkpoint saved")
			checkpoint = torch.load("checkpoint.pth.tar")
		else:
			checkpoint = torch.load(max(list_checkpoints, key=os.path.getctime))
	try:
		net.load_state_dict(checkpoint['network_state_dict'])
	except:
		### WHEN YOU HAVE MIXED SPARSE AND DENSE PARAMETERS
		### LOAD PARAMETERS MANUALLY - for partial bayesian 
		return checkpoint
	epoch_start =0
	if cont_run:
		epoch_start = checkpoint['epoch']
	return net, epoch_start

class FocalLoss(nn.Module):
	def __init__(self, alpha=1, gamma=2, reduction='mean'):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, input, target):
		ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=0)
		pt = torch.exp(-ce_loss)
		focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		else:
			return focal_loss

def get_loss_func(loss_func: str, dataset=""):
	try:
		if loss_func == 'mse':
			return nn.MSELoss()
		elif loss_func == 'ce':
			if dataset == 'isic': 
				weight = torch.tensor([1, 1], dtype=torch.float32).to(torch.device('cuda'))
				return nn.CrossEntropyLoss(weight=weight)
			if dataset == 'lidc': 
				weight = torch.tensor([1, 10], dtype=torch.float32).to(torch.device('cuda'))
				return nn.CrossEntropyLoss(weight=weight)
			return nn.CrossEntropyLoss()
		elif loss_func == 'bce':
			return nn.BCELoss()
		elif loss_func == 'bceLogits':
			if dataset.upper() == "CHESTMNIST":
				#pos_weight computes as follows
				#num_positives = targets.sum(dim=0)
				#num_negatives = (1 - targets).sum(dim=0)
				#total_samples = num_positives + num_negatives
				#pos_weight = num_negatives / num_positives
				#neg_weight = num_positives / num_negatives
				pos_weight = torch.tensor([8.2698, 37.5447, 7.1456, 4.6965, 18.7996, 15.8037, 91.6983, 19.5996, 22.4410, 53.3172, 43.0727, 60.9696, 29.5627, 533.1190], device='cuda')
				return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
			return nn.BCEWithLogitsLoss()
		elif loss_func == 'nll':
			return nn.NLLLoss()
		elif loss_func == 'focal':
			return FocalLoss()
		elif loss_func == 'dice':
			return DiceLoss()
	except: 
		raise Exception(f"{loss_func} LOSS FUNCTION NOT IMPLEMENTED")

def get_lr_scheduler(scheduler_info: dict, optimizer:torch.optim):
	scheduler_type = scheduler_info["scheduler_type"]
	try:
		gamma = scheduler_info["gamma"]
		if scheduler_type == "": 
			return None 
		elif scheduler_type == "step":
			step_size = scheduler_info["step_size"]
			return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
		elif scheduler_type.upper() == "REDUCELRONPLATEAU":
			return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=gamma, min_lr=1e-6)
	except:
		raise Exception(f"{scheduler_type} Scheduler is not implemented, \
				  choose one of the following options: ['', step, ReduceLROnPlateau]")

def get_optimizer(model, opt_type='adam', lr=0.1, weight_decay=0, sensitivity=None):
	if opt_type == 'adam':
		return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif opt_type == 'sparseAdam':
		return torch.optim.SparseAdam(model.parameters(), lr=lr)
	elif opt_type == 'sgd':
		if torch.__version__[0] == '2':
			return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, foreach=False, weight_decay=weight_decay)
		else: 
			return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
	elif opt_type == 'adagrad':
		return torch.optim.Adagrad(model.parameters(), lr=lr, foreach=False, weight_decay=weight_decay)
	else: 
		raise Exception(f"{opt_type} OPTIMIZER NOT IMPLEMENTED")

"""==================================================="""
"""================ General Utilities ================"""
"""==================================================="""
def get_logging_keys(net, task, run_val=False, test=False, dataset=''):
	keys_mapping = {
		"deterministic": {
			"regression": ["RMSE"],
			"classification": ["Accuracy"],
			"segmentation": ["Dice"],
			"segmentation2": ["IoU"],
		},
		"bayesian": {
			"regression": ["KL", "RMSE", "Std"],
			"classification": ["KL", "Accuracy", "Brier Score"],
			"segmentation": ["KL", "Dice", "Brier Score"],
			"segmentation2": ["KL", "IoU", "Brier Score"],
		},
		"partial_bayesian": {
			"regression": ["KL", "RMSE", "Std"],
			"classification": ["KL", "Accuracy", "Brier Score"],
			"segmentation": ["KL", "Dice", "Brier Score"],
			"segmentation2": ["KL", "IoU", "Brier Score"],
		}
	}

	if dataset == 'isic':
		task = "segmentation2"
	train_keys = [f"Train {key}" for key in keys_mapping.get(net, {}).get(task, [])]
	test_keys = []
	if run_val or test:
		keys_suffix = "Test" if test else "Val"
		test_keys = [f"{keys_suffix} Time", f"{keys_suffix} Loss"]
		test_keys += [f"{keys_suffix} {key}" for key in keys_mapping.get(net, {}).get(task, [])]

	if test and "segmentation" not in task:
		if task == "regression":
			logging_keys = ["Y", "Y_pred"] + test_keys
		elif task == "classification":
			if "MNIST" in dataset.upper() and len(dataset) > len("mnist"):
				logging_keys = ["Y_conf"] + test_keys + ["Test Entropy", "TP", "TN", "FP", "FN"]
			else:
				logging_keys = ["Y", "Y_pred", "Y_conf"] + test_keys + ["Test Entropy", "Test Brier Score"]
	elif test and "segmentation" in task:
		logging_keys = test_keys + ["Test Entropy"]
		if "Test Brier Score" not in logging_keys:
			logging_keys.append("Test Brier Score")
	else:
		logging_keys = ["Kfold", "Epoch", "Train Time", "Train Loss"] + train_keys + test_keys

	metrics = {key: [] for key in logging_keys}

	if run_val or not test:
		best_metric_prefix = "Best Val" if run_val else "Best Train"
		if task == "classification":
			metrics[f'{best_metric_prefix} Accuracy'] = 0
		elif task == 'regression':
			metrics[f'{best_metric_prefix} RMSE'] = 1e4
		elif task == 'segmentation':
			metrics[f'{best_metric_prefix} Dice'] = 0
		if task == 'segmentation2':
			metrics[f'{best_metric_prefix} IoU'] = 0

	return metrics, logging_keys


class CSVlogger():
	def __init__(self, log_name, header_names):
		self.logging_keys = header_names
		self.log_name     = log_name
	   
		with open(log_name,"a") as csv_file:
			writer = csv.writer(csv_file, delimiter=",")
			writer.writerow(self.logging_keys)
	def write(self, inputs):
		with open(self.log_name,"a") as csv_file:
			writer = csv.writer(csv_file, delimiter=",")
			writer.writerow(inputs)

def read_config(out_dir: str, file='config.json'):
	file = Path(out_dir) / file
	with open(file, 'r') as f:
		opts = json.load(f)
	return opts

def log_runs(opts: dict, out_dir = None, file='log_runs.csv'):

	run = opts["Paths"]["save_path"]
	log_keys = ["run"]
	log_values = [Path(run).resolve().name]

	if out_dir is None: 
		out_dir = Path(__file__).resolve().parent
	
	opts = parse_json(opts, net_type=opts["Network"]["Basic Setup"]["network_type"])
	log_keys += list(opts.keys())    
	log_values += list(opts.values()) 

	log_file_path = Path(out_dir) / file

	if not log_file_path.exists():
		with open(log_file_path, 'w') as log_file:
			log_file.write(','.join(log_keys))

	with open(log_file_path, 'a') as log_file:
		log_file.write("\n")
		for val in log_values:
			log_file.write(str(val) + ',')
		
def parse_json(data: json, keys={}, net_type="deterministic", empty=False):
	empty_values = False
	if "partial" not in net_type: 
		empty_values = True

	for key, value in data.items(): 
		if isinstance(value, dict):
			if "partial" in key and empty_values: 
				empty = True 
			parse_json(value, keys=keys, empty=empty)
		else:
			if empty == True: 
				value = "" 
			keys.update({key: value})
	return keys

def save_pred(input, target, output, epoch, output_path, dataset, train=True, *args):
	num_samples = 5 if train else 1
	tar_idx = 0 if target.shape[1] == 1 else 1
	posterior = False
	if output.shape != target.shape:
		posterior = True
		variance = output.var(dim=0)
		output = output.mean(dim=0)		

	cols = 5 if args else 4
	cols += 1 if posterior else 0
	_, axarr = plt.subplots(num_samples, cols, gridspec_kw = {'wspace':0.02, 'hspace':0})
	for i in range(num_samples):	
		if num_samples != 1:
			i1, i2, i3, i4, i5, i6  = (i, 0), (i, 1), (i, 2), (i, 3), (i, 4), (i, 5)
		else:
			i1, i2, i3, i4, i5, i6 = 0, 1, 2, 3, 4, 5
		
		img = input[i].squeeze(0).cpu().numpy()
		cmap_in = 'gray'
		if dataset.upper() == 'ISIC':
			img = isic.invert_img_norm(input[i])
			cmap_in = 'viridis' 
		else:
			label = target[i][tar_idx].squeeze(0).cpu().numpy()
			conf = output[i][tar_idx].detach().squeeze(0).cpu().numpy() 
			pred = output[i][tar_idx].detach().squeeze(0).cpu().numpy() > 0.5
			cmap = 'gray'
		axarr[i1].imshow(img, cmap=cmap_in, origin="lower")
		axarr[i2].imshow(label, cmap=cmap, origin="lower")	
		axarr[i3].imshow(pred, cmap=cmap, origin="lower")
		if i == 0:
			axarr[i1].title.set_text('input')
			axarr[i2].title.set_text('target')
			axarr[i3].title.set_text('pred')	
		if cols > 3:
			axarr[i4].imshow(conf, cmap='gray', origin="lower")
			if args:
				uncert_gt = args[0][i].detach().squeeze(0).cpu().numpy()
				axarr[i5].imshow(uncert_gt, cmap='seismic', origin="lower")
			if i == 0:
				axarr[i4].title.set_text('conf')
				if args:
					axarr[i5].title.set_text('gt_u')
			if posterior:
				uncert_idx = i6 if args else i5
				uncert_pred = variance[i][tar_idx].detach().squeeze(0).cpu().numpy()
				axarr[uncert_idx].imshow(uncert_pred, cmap='seismic', origin="lower")
				if i == 0:
					axarr[uncert_idx].title.set_text('pred_u')					

	for i in range(num_samples):
		for j in range(cols): 
			idx = (i, j) if num_samples != 1 else j
			axarr[idx].axis('off') 
	os.chdir(output_path)
	plt.savefig(f'pred{epoch}.pdf', format="pdf", dpi=1200)
	plt.close('all')

def append_to_hdf5(file_name, output_path, name, img, target, pred, *args):
	name = str(name)
	file_path = os.path.join(output_path, file_name)
	with h5py.File(file_path, 'a') as hf:
		group = hf.create_group(name)
		group.create_dataset('image', data=img.squeeze(0).squeeze(0).detach().cpu())
		group.create_dataset('mask', data=target.squeeze(0).detach().cpu())
		group.create_dataset('pred', data=pred.detach().cpu())
		if args:
			group.create_dataset('gt_uncert', data=args[0].detach().cpu().squeeze(0).squeeze(0))
			group.create_dataset('gt_mean', data=args[1].detach().cpu().squeeze(0).squeeze(0))

