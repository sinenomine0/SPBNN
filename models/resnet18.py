import models.maskedtensor as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
from .deterministic import get_layer
from .bayeslayers import LinearReparameterization, Conv2dReparameterization

def initialize_batchNorm(feature_size: int, init_weight:torch.Tensor, init_bias: torch.Tensor, turn_off: bool=True):
	tmp = nn.BatchNorm2d(feature_size, track_running_stats=turn_off)
	tmp.weight = init_weight
	tmp.bias = init_bias
	return tmp

def get_size(size, pad, dilate, kernel, stride, avgpool=False): 
	"""
	source: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
	source: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
	"""
	#shape = [height + 2*padding - dilation x (kernel_size-1) - 1]/stride  + 1
	pad = pad[0] if isinstance(pad, tuple) else pad 
	dilate = dilate[0] if isinstance(dilate, tuple) else dilate
	kernel = kernel[0] if isinstance(kernel, tuple) else kernel
	stride = stride[0] if isinstance(stride, tuple) else stride
	if avgpool:
		tmp = size + (2*pad) - kernel
	else:
		tmp = size + (2*pad) - dilate*(kernel-1) - 1
	out = tmp/stride 
	return math.floor(out+1)



class BayesConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, bias: bool=False, 
				 kernel_size: int=3, stride: int=1, padding: int=0, prior_mu: float=0., 
				 prior_var: float=1., posterior_mu_init: float=0., posterior_rho_init: float=-3.,
				 sparse: dict={}, nonlinearity: nn.Module=nn.ReLU(), masks: list=[], 
				 init_params: list=[], fit_variance_only: bool=False, test: bool=False):
		super().__init__()
		layers =[]
		bias_skip = 1 if bias else 0
		layers.append(Conv2dReparameterization(in_channels=in_channels, out_channels=out_channels, bias=bias,
											   kernel_size=kernel_size, stride=2, padding=padding,
											   prior_mean=prior_mu, prior_variance=prior_var, sparse=sparse,
											   posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init,
											   mask=masks[0:bias_skip+1], init_param=init_params[0:bias_skip+1], 
											   fit_variance_only=fit_variance_only, test=test))
		if init_params:
			layers.append(initialize_batchNorm(out_channels, init_params[bias_skip+1], init_params[bias_skip+2], turn_off= not fit_variance_only))
		else:
			layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nonlinearity)
		bias_skip += 3
		layers.append(Conv2dReparameterization(in_channels=out_channels, out_channels=out_channels, bias=bias,
											   kernel_size=kernel_size, stride=1, padding=padding,
											   prior_mean=prior_mu, prior_variance=prior_var, sparse=sparse,
											   posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init,
											   mask=masks[bias_skip:-2], init_param=init_params[bias_skip:-2],
											   fit_variance_only=fit_variance_only, test=test))
		if init_params:
			layers.append(initialize_batchNorm(out_channels, init_params[-2], init_params[-1], turn_off= not fit_variance_only))
		else:
			layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nonlinearity)

		self.layers = nn.Sequential(*layers)   
	
	def forward(self, x, return_kl=True):    
		kl_sum = 0
		for l in self.layers:
			name = l.__class__.__name__
			if "Reparameterization" in name or "Bayes" in name:
				if return_kl:
					x, kl = l(x)
					kl_sum = kl_sum + kl
				else: 
					x = l(x, return_kl=return_kl)
			else:
				x = l(x)
		if return_kl:
			return x, kl
		return x


class BayesResBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, bias: bool=False, 
				 kernel_size: int=3, stride: int=1, padding: int=0, prior_mu: float=0., 
				 prior_var: float=1., posterior_mu_init: float=0., posterior_rho_init: float=-3.,
				 sparse: dict={}, nonlinearity: nn.Module=nn.ReLU(), masks: list=[], 
				 init_params: list=[], fit_variance_only: bool=False, test: bool=False):
		super().__init__()
		self.convblock = BayesConvBlock(in_channels, out_channels, bias, kernel_size, stride, padding, prior_mu, 
				 prior_var, posterior_mu_init, posterior_rho_init, sparse, nonlinearity, masks[0:-3], init_params[0:-3], 
				 fit_variance_only, test)
		self.skip = None
		# If the input and output channels are different, use a 1x1 convolution to match them
		if stride != 1 or in_channels != out_channels:
			layers =[]
			layers.append(Conv2dReparameterization(in_channels=in_channels, out_channels=out_channels, bias=False,
										 kernel_size=1, stride=2, padding=0, prior_mean=prior_mu, 
										 prior_variance=prior_var, posterior_mu_init=posterior_mu_init, 
										 posterior_rho_init=posterior_rho_init, sparse=sparse, mask=masks[-3:-3], 
										 init_param=init_params[-3:-3], fit_variance_only=fit_variance_only))
			if init_params:
				layers.append(initialize_batchNorm(out_channels, init_params[-2], init_params[-1],  turn_off= not fit_variance_only))
			else:
				layers.append(nn.BatchNorm2d(out_channels))
			layers.append(nonlinearity)
			self.skip = nn.Sequential(*layers)

	def forward(self, x, return_kl=True):
		kl_sum = 0
		conv = self.convblock(x, return_kl)
		if return_kl:
			out, kl = conv
			kl_sum = kl_sum + kl
		else:
			out = conv
		if self.skip is not None:			
			for l in self.skip:
				if "Reparameterization" in l.__class__.__name__:
					if return_kl:
						x, kl = l(x)
						kl_sum = kl_sum + kl
					else: 
						x = l(x, return_kl=return_kl)
				else:
					x = l(x)
			if return_kl:
				out = out + x
				return out, kl_sum	
		return out
	
	
class BayesResNet18(nn.Module):
	def __init__(self, input_size: int, feature_size: list, output_size: int, add_bias: bool=True,
				 kernel_size: int=3, stride: int=1, padding: int=0, pooling_size: int=2,
				 prior_mu: float=0., prior_variance: float=1., posterior_mu: float=0., 
				 posterior_rho: float=-3., sparse: dict={}, masks: list=[], init_params: list=[],
				 fit_variance_only: bool=False, nonlinearity: str="relu", test: bool=False):
		super().__init__()
		nonlinearity = utils.get_nonlinearity(nonlinearity=nonlinearity)
		layers= []
		idx_skip, bNidx = 0, 0
		track_size = input_size
		for i, l in enumerate(feature_size):
			layer_masks, layer_params = [], []
			if masks!=[]:
				if i == 0:
					if add_bias:
						i0, ifinal = idx_skip, idx_skip+2
						bNidx = idx_skip+2 #batchNorm
						idx_skip += 4
					else: 
						i0, ifinal = i, i
						bNidx = i+1
						idx_skip += 3
				else:
					if add_bias: 
						i0, ifinal = idx_skip, idx_skip+11
						idx_skip += 11
					else: 
						i0, ifinal = idx_skip, idx_skip+9
						idx_skip += 9
				if i == len(feature_size)-1:
					i0 = -4 if add_bias else -2
					ifinal = None
				layer_masks = masks[i0:ifinal]
				layer_params = init_params[i0:ifinal]

			if i == len(feature_size)-1:
				layers.append(nn.AvgPool2d(kernel_size=(7,7), stride=(1,1)))
				track_size = get_size(track_size, pad=0, dilate=0, kernel=7, stride=1, avgpool=True)
				layers.append(nn.Flatten())
				layers.append(LinearReparameterization(track_size*feature_size[-1], 1000, prior_mu, prior_variance,
										   posterior_mu, posterior_rho, add_bias, sparse, 
										   layer_masks[0:2], layer_params[0:2], fit_variance_only, test=test))
				layers.append(LinearReparameterization(1000, output_size, prior_mu, prior_variance,
										posterior_mu, posterior_rho, add_bias, sparse, layer_masks[2:], layer_params[2:],
										fit_variance_only, test=test))
			elif i == 0:
				layers.append(Conv2dReparameterization(in_channels=l, out_channels=feature_size[i+1], bias=add_bias,
													kernel_size=7, stride=2, padding=3, prior_mean=prior_mu, 
													prior_variance=prior_variance, posterior_mu_init=posterior_mu, 
													posterior_rho_init=posterior_rho, sparse=sparse,
													mask=layer_masks, init_param=layer_params, 
													fit_variance_only=fit_variance_only, test=test))
				if init_params:
					#load learned params
					layers.append(initialize_batchNorm(feature_size[i+1], init_params[bNidx], init_params[bNidx+1], turn_off= not fit_variance_only))
				else:
					layers.append(nn.BatchNorm2d(feature_size[i+1]))
				layers.append(nonlinearity)
				layers.append(nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)))

				track_size = get_size(track_size, pad=3, dilate=1, kernel=7, stride=2) #first conv layer
				track_size = get_size(track_size, pad=1, dilate=1, kernel=3, stride=2) #max pool layer
			else:
				layers.append(BayesResBlock(l, feature_size[i+1], add_bias, kernel_size, stride, padding, prior_mu, 
											prior_variance, posterior_mu, posterior_rho, sparse, nonlinearity, 
											layer_masks, layer_params, fit_variance_only, test=test))
				track_size = get_size(track_size, pad=padding, dilate=1, kernel=kernel_size, stride=2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x, return_kl=True, nll_loss=False):
		kl_sum = 0
		for l in self.layers:
			name = l.__class__.__name__
			if "Reparameterization" in name or "Bayes" in name:
				if return_kl:
					x, kl = l(x)
					kl_sum = kl_sum + kl
				else: 
					x = l(x, return_kl=return_kl)
			else:
				x = l(x)
		if nll_loss:
			x = F.log_softmax(x, dim=-1)
		if return_kl:
			return x, kl_sum
		return x


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, bias: bool, 
			  	kernel_size: int, stride: int=1, padding: int=0, 
				sparse: dict={}, nonlinearity: nn.Module=nn.ReLU()):
		super().__init__()
		layers = []
		layers.append(get_layer(input=in_channels, output=out_channels, sparse=sparse["sparse"], 
							sparse_params=sparse, layer_type='conv', kernel=kernel_size, 
							stride=2, padding=padding, bias=bias))   
		layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nonlinearity)
		layers.append(get_layer(input=out_channels, output=out_channels, sparse=sparse["sparse"], 
							sparse_params=sparse, layer_type='conv', kernel=kernel_size, 
							stride=1, padding=padding, bias=bias))   
		layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nonlinearity) 
	
		self.layers = nn.Sequential(*layers)

	def forward(self, x):    
		x = self.layers(x)    
		return x
	

class ResBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool,
			  stride: int=1, padding: int=0, sparse: dict={}, nonlinearity: nn.Module=nn.ReLU()):
		super().__init__()
		self.convblock = ConvBlock(in_channels, out_channels, bias, kernel_size, stride, padding, sparse, nonlinearity)
		self.skip = nn.Sequential()
		# If the input and output channels are different, use a 1x1 convolution to match them
		if stride != 1 or in_channels != out_channels:
			self.skip = nn.Sequential(
			get_layer(input=in_channels, output=out_channels, sparse=sparse["sparse"], 
						sparse_params=sparse, layer_type='conv', kernel=1, stride=2, 
						padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nonlinearity
			)

	def forward(self, x):
		out = self.convblock(x)
		out = out + self.skip(x)
		return out
	

class ResNet18(nn.Module):
	def __init__(self, input_size: int, feature_size: list, output_size: int, add_bias: bool,
				 kernel_size: int, stride: int, pooling_size: int=2, padding: int=0,
				 sparse: dict={}, nonlinearity: str='relu'):
		super().__init__()
		nonlinearity = utils.get_nonlinearity(nonlinearity=nonlinearity)
		layers= []
		track_size = input_size
		for i, l in enumerate(feature_size):
			if i == len(feature_size)-1:
				layers.append(nn.AvgPool2d(kernel_size=(7,7), stride=(1,1)))
				track_size = get_size(track_size, pad=0, dilate=0, kernel=7, stride=1, avgpool=True)
				layers.append(nn.Flatten())
				layers.append(nn.Linear(track_size*feature_size[-1], out_features=1000, bias=add_bias))
				layers.append(nn.Linear(1000, output_size, bias=add_bias))
			elif i == 0:
				layers.append(get_layer(input=l, output=feature_size[i+1], sparse=sparse["sparse"], 
									sparse_params=sparse, layer_type='conv', kernel=7, 
									stride=2, padding=3, bias=add_bias))   
				layers.append(nn.BatchNorm2d(feature_size[i+1]))
				layers.append(nonlinearity)
				layers.append(nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)))
				
				track_size = get_size(track_size, pad=3, dilate=1, kernel=7, stride=2) #first conv layer
				track_size = get_size(track_size, pad=1, dilate=1, kernel=3, stride=2) #max pool layer
			else:
				layers.append(ResBlock(in_channels=l, out_channels=feature_size[i+1], bias=add_bias, 
						   		kernel_size=kernel_size, stride=stride, padding=padding, 
								sparse=sparse, nonlinearity=nonlinearity))
				track_size = get_size(track_size, pad=padding, dilate=1, kernel=kernel_size, stride=2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x, nll_loss=False):
		x = self.layers(x)
		if nll_loss:
			x = F.log_softmax(x, dim=-1)
		return x