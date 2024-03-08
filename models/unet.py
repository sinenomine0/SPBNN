import torch
import torch.nn as nn
import torch.nn.functional as F
from .bayeslayers import Conv2dReparameterization


def initialize_batchNorm(feature_size: int, init_weight:torch.Tensor, init_bias: torch.Tensor, turn_off: bool=True):
	tmp = nn.BatchNorm2d(feature_size, track_running_stats=turn_off)
	tmp.weight = init_weight
	tmp.bias = init_bias
	return tmp


class DownConvBlock(nn.Module):
	"""
	A block of three convolutional layers where each layer is followed by a non-linear activation function
	Between each block we add a pooling operation.
	"""
	def __init__(self, in_channels, out_channels, padding, pool=True, dropout=False, dropout_rate=0, batch_norm=True):
		super().__init__()
		layers = []

		if pool:
			layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

		for _ in range(3):
			layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=int(padding)))
			if batch_norm:
				layers.append(nn.BatchNorm2d(num_features=out_channels))
			else: 
				layers.append(nn.InstanceNorm2d(num_features=out_channels))
			layers.append(nn.ReLU(inplace=True))

			in_channels = out_channels

		if dropout:
			layers.append(nn.Dropout(p=dropout_rate))
	   
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class UpConvBlock(nn.Module):
	"""
	A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
	If bilinear is set to true upsampling is used, if false - transpose is used - however cannot be used for partial bayes
	"""
	def __init__(self, in_channels: int, out_channels: int, padding: int=0, interpolate: bool=True, batch_norm: bool=True):
		super().__init__()
		self.interpolate = interpolate
		in_ = in_channels+out_channels
		if not interpolate:
			self.upconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
			in_ = out_channels*2
		self.convblock = DownConvBlock(in_, out_channels, padding, pool=False, batch_norm=batch_norm)

	def forward(self, x, bridge):
		if self.interpolate:
			up = F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
		else:
			up = self.upconv_layer(x)      
		assert up.shape[3] == bridge.shape[3]
		out = torch.cat([up, bridge], 1)
		out =  self.convblock(out)
		return out


class UNet(nn.Module):
	"""
	A UNet (https://arxiv.org/abs/1505.04597) implementation.
	in_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
	output_size: the number of classes to predict
	feature_size: list with the amount of filters per layer
	apply_classifier: boolean to apply last layer or not (not used in Probabilistic UNet)
	padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
	"""

	def __init__(self, in_channels, output_size, feature_size, apply_classifier=True, 
						padding: bool=True, batch_norm: bool=True, interpolate: bool=True):
		super().__init__()
		self.in_channels = in_channels
		self.output_size = output_size
		self.feature_size = feature_size
		self.padding = padding
		self.activation_maps = []
		self.apply_classifier = apply_classifier
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()

		for i in range(len(self.feature_size)):
			input = self.in_channels if i == 0 else output
			output = self.feature_size[i]
			pool = False if i == 0 else True
			self.encoder.append(DownConvBlock(input, output, padding, pool=pool, batch_norm=batch_norm))

		for output, input in zip(feature_size[:-1][::-1], feature_size[::-1]):
			self.decoder.append(UpConvBlock(input, output, padding, interpolate, batch_norm=batch_norm))

		if self.apply_classifier:
			self.classifier = nn.Conv2d(output, output_size, kernel_size=1)

	def forward(self, x, **kwargs):
		blocks = []
		for i, down in enumerate(self.encoder):
			x = down(x)
			if i != len(self.encoder)-1:
				blocks.append(x)

		for i, up in enumerate(self.decoder):
			x = up(x, blocks[-i-1])

		del blocks

		if self.apply_classifier:
			x =  self.classifier(x)
		
		return x
	

class BayesDownConvBlock(nn.Module):
	"""
	A block of three convolutional layers where each layer is followed by a non-linear activation function
	Between each block we add a pooling operation.
	"""
	def __init__(self, in_channels, out_channels, padding, add_bias: bool=True,
			  prior_mu: float=0., prior_variance: float=1., posterior_mu: float=0.,
			  posterior_rho: float=-3, pool: bool=True, batch_norm: bool=True,
			  sparse: dict={}, masks: list=[], init_params: list=[],
			  fit_variance_only: bool=False, test: bool=False):
		super().__init__()
		layers = []

		if pool:
			layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

		idx_skip = 0
		for _ in range(3):
			num_ = 2 if add_bias else 1
			block_masks, block_params = [], []
			if masks!=[]:				
				block_masks = masks[idx_skip:idx_skip+num_]
				block_params = init_params[idx_skip:idx_skip+num_]
				idx_skip += num_
			layers.append(Conv2dReparameterization(in_channels, out_channels, kernel_size=3, stride=1, padding=int(padding), prior_mean=prior_mu, 
													prior_variance=prior_variance, posterior_mu_init=posterior_mu, posterior_rho_init=posterior_rho,
													sparse=sparse, mask=block_masks, init_param=block_params, 
													fit_variance_only=fit_variance_only, test=test))
			if batch_norm:
				if init_params:
						layers.append(initialize_batchNorm(out_channels, init_params[idx_skip], init_params[idx_skip+1], turn_off= not fit_variance_only))
						idx_skip += 2
				else:
					layers.append(nn.BatchNorm2d(num_features=out_channels))
			else:
				layers.append(nn.InstanceNorm2d(num_features=out_channels))
			layers.append(nn.ReLU(inplace=True))

			in_channels = out_channels
	   
		self.layers = nn.Sequential(*layers)

	def forward(self, x, return_kl=True):
		kl_sum = 0
		for l in self.layers:
			if "Reparameterization" in l.__class__.__name__:
				if return_kl:
					x, kl = l(x)
					kl_sum = kl_sum + kl
				else: 
					x = l(x, return_kl=return_kl)
			else:
				x = l(x)
		if return_kl:
			return x, kl_sum        
		return x


class BayesUpConvBlock(nn.Module):
	"""
	A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
	If bilinear is set to true - bilinear interpolation is done - transpose conv custom bayes layer is not implemented
	"""
	def __init__(self, in_channels, out_channels, padding, add_bias: bool=True,
			  prior_mu: float=0., prior_variance: float=1., posterior_mu: float=0.,
			  posterior_rho: float=-3, sparse: dict={}, batch_norm: bool=True, 
			  masks: list=[], init_params: list=[], fit_variance_only: bool=False,
			  test: bool=False, interpolate: bool=True):
		super().__init__()

		self.interpolate = interpolate
		in_ = in_channels+out_channels

		block_masks, block_params = masks, init_params
		self.convblock = BayesDownConvBlock(in_, out_channels, padding, add_bias, prior_mu, prior_variance, posterior_mu, 
											 posterior_rho, pool=False, batch_norm=batch_norm, sparse=sparse, masks=block_masks, 
											 init_params=block_params, test=test, fit_variance_only=fit_variance_only)

	def forward(self, x, bridge, return_kl=True):
		kl_sum = 0
		up = F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
		assert up.shape[3] == bridge.shape[3]
		out = torch.cat([up, bridge], 1)
		if return_kl:
			out, kl = self.convblock(out)
			kl_sum = kl_sum + kl
		else:
			out =  self.convblock(out, return_kl=return_kl)

		if return_kl:
			return out, kl_sum
		return out


class BayesUNet(nn.Module):
	"""
	A UNet (https://arxiv.org/abs/1505.04597) implementation.
	in_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
	output_size: the number of classes to predict
	feature_size: list with the amount of filters per layer
	apply_classifier: boolean to apply last layer or not (not used in Probabilistic UNet)
	padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
	"""

	def __init__(self, in_channels: int, output_size: int, feature_size: list, add_bias: bool=True,
				 prior_mu: float=0., prior_variance: float=1., posterior_mu: float=0.,
				 posterior_rho: float=-3, apply_classifier=True, padding=True, batch_norm=True, 
				sparse: dict={}, masks: list=[], init_params: list=[], fit_variance_only: bool=False,
				test: bool=False, interpolate: bool=True):
		super().__init__()
		self.in_channels = in_channels
		self.output_size = output_size
		self.feature_size = feature_size
		self.apply_classifier = apply_classifier
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()

		idx_skip = 0
		num_ = 12 if (add_bias and batch_norm) else 6
		for i in range(len(self.feature_size)):
			block_masks, block_params = [], []
			if masks!=[]:				
				block_masks = masks[idx_skip:idx_skip+num_]
				block_params = init_params[idx_skip:idx_skip+num_]
				idx_skip += num_
			input = self.in_channels if i == 0 else output
			output = self.feature_size[i]
			pool = False if i == 0 else True
			self.encoder.append(BayesDownConvBlock(input, output, padding, add_bias, prior_mu, prior_variance, 
										  posterior_mu, posterior_rho, pool, batch_norm, sparse, block_masks, 
										  block_params, fit_variance_only, test))

		num_ = num_ + 2 if not interpolate else num_	
		for output, input in zip(feature_size[:-1][::-1], feature_size[::-1]):
			block_masks, block_params = [], []
			if masks!=[]:				
				block_masks = masks[idx_skip:idx_skip+num_]
				block_params = init_params[idx_skip:idx_skip+num_]
				idx_skip += num_
			self.decoder.append(BayesUpConvBlock(input, output, padding, add_bias, prior_mu, prior_variance, 
										posterior_mu, posterior_rho, sparse, batch_norm, block_masks, 
										block_params, fit_variance_only, test, interpolate))

		if self.apply_classifier:
			num_ = -2 if add_bias else -1
			block_masks, block_params = [], []
			if masks!=[]:				
				block_masks, block_params = masks[num_:], init_params[num_:]
			self.classifier = Conv2dReparameterization(in_channels=output, out_channels=output_size, kernel_size=1, 
													   prior_mean=prior_mu, prior_variance=prior_variance, 
													   posterior_mu_init=posterior_mu, posterior_rho_init=posterior_rho, 
													   sparse=sparse, mask=block_masks, init_param=block_params, test=test,
													   fit_variance_only=fit_variance_only)

	def forward(self, x, return_kl=True, **kwargs):
		kl_sum = 0
		blocks = []
		for i, down in enumerate(self.encoder):
			if return_kl:
				x, kl = down(x)
				kl_sum = kl_sum + kl
			else:
				x = down(x, return_kl=return_kl)
			if i != len(self.encoder)-1:
				blocks.append(x)

		for i, up in enumerate(self.decoder):
			if return_kl:
				x, kl = up(x, blocks[-i-1])
				kl_sum = kl_sum + kl
			else:
				x = up(x, blocks[-i-1], return_kl=return_kl)

		del blocks

		if self.apply_classifier:
			if return_kl:
				x, kl =  self.classifier(x)
				kl_sum = kl_sum + kl
			else:
				x =  self.classifier(x, return_kl=return_kl)

		if return_kl:
			return x, kl_sum        
		return x
