import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class MaskedLinearReparameterizationFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, weight_sigma, weight_mask,
			 	bias, bias_sigma, bias_mask):
		
		ctx.save_for_backward(input, weight, weight_sigma, bias, bias_sigma)
		#tensors that do not require gradients are saved as is
		ctx.weight_mask = weight_mask 
		ctx.bias_mask = bias_mask
		
		if bias is not None:
			return F.linear(input, weight+weight_sigma, bias+bias_sigma)
		else:
			return F.linear(input, weight+weight_sigma, bias)
			
	@staticmethod
	def backward(ctx, grad_output):
		input, weight, weight_sigma, bias, bias_sigma = ctx.saved_tensors
		weight_mask = ctx.weight_mask
		bias_mask = ctx.bias_mask

		grad_input = grad_weight = grad_weight_sigma = grad_bias = grad_bias_sigma = None

		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(weight + weight_sigma)
		if ctx.needs_input_grad[1]:
			grad_weight = grad_output.t().mm(input)
		if ctx.needs_input_grad[2]:
			grad_weight_sigma = grad_output.t().mm(input) * weight_mask
			grad_weight_sigma = grad_weight_sigma.to_sparse()
		if ctx.needs_input_grad[4]:
			grad_bias = grad_output.sum(0)
		if ctx.needs_input_grad[5]:
			grad_bias_sigma = grad_output.sum(0) * bias_mask
			grad_bias_sigma = grad_bias_sigma.to_sparse()

		return grad_input, grad_weight, grad_weight_sigma, None, grad_bias, grad_bias_sigma, None


class MaskedConv2dReparameterizationFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, weight_sigma, weight_mask, 
			 bias, bias_sigma, bias_mask, stride, padding, groups):

		ctx.save_for_backward(input, weight, weight_sigma, bias, bias_sigma)
		ctx.weight_mask = weight_mask
		ctx.bias_mask = bias_mask
		ctx.stride = stride
		ctx.padding = padding
		ctx.groups = groups

		if bias is not None:
			return F.conv2d(input, weight+weight_sigma, bias+bias_sigma, stride=stride, padding=padding, groups=groups)
		else:
			return F.conv2d(input, weight+weight_sigma, bias, stride=stride, padding=padding, groups=groups)

	@staticmethod
	def backward(ctx, grad_output):
		input, weight, weight_sigma, bias, bias_sigma = ctx.saved_tensors
		weight_mask = ctx.weight_mask #idx#3 
		bias_mask = ctx.bias_mask #idx#6
		stride = ctx.stride #idx#7
		padding = ctx.padding #idx#8
		groups = ctx.groups #idx#9
		# Compute the gradients of the input, weight, and bias tensors
		grad_input = grad_weight = grad_bias = grad_weight_sigma = grad_bias_sigma = None
		if ctx.needs_input_grad[0]:
			grad_input = torch.nn.grad.conv2d_input(input.shape, weight + weight_sigma, grad_output, stride=stride, padding=padding, groups=groups)
		if ctx.needs_input_grad[1]:
			grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
		if ctx.needs_input_grad[2]:
			grad_weight_sigma = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
			grad_weight_sigma = grad_weight_sigma * weight_mask
			grad_weight_sigma = grad_weight_sigma.to_sparse()
		if bias is not None and ctx.needs_input_grad[4]:
			grad_bias = grad_output.sum((0, 2, 3))
		if bias is not None and ctx.needs_input_grad[5]:
			grad_bias_sigma = grad_output.sum((0, 2, 3)) * bias_mask
			grad_bias_sigma = grad_bias_sigma.to_sparse()
		return grad_input, grad_weight, grad_weight_sigma, None, grad_bias, grad_bias_sigma, None, None, None, None
	
class MaskedLinearFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias, mask):
		# Save the input, weight, bias, and mask tensors for backward
		mask = mask.to(weight.device)
		ctx.save_for_backward(input, weight, bias)
		ctx.mask = mask
		# Perform the linear operation using the masked weight tensor
		return F.linear(input, weight, bias)

	@staticmethod
	def backward(ctx, grad_output):
		# Retrieve the saved tensors from forward
		input, weight, bias = ctx.saved_tensors
		mask = ctx.mask
		# Compute the gradients of the input, weight, and bias tensors
		grad_input = grad_weight = grad_bias = None
		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(weight * mask)
		if ctx.needs_input_grad[1]:
			grad_weight = grad_output.t().mm(input)
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0)
			# Apply the mask to the weight gradient to make it sparse
		grad_weight = (grad_weight * mask)
		grad_weight = grad_weight.to_sparse()
		return grad_input, grad_weight, grad_bias 


class MaskedLinear(nn.Module):
	def __init__(self, in_features: int, out_features: int, 
		mask = None, bias=True, rate=0) -> None:
		super().__init__()
		self.mask = mask
		self.bias = None
		weight = torch.nn.init.kaiming_normal_(torch.Tensor(out_features, in_features))
		if self.mask is None: 
			self.mask = utils.get_sparse_mask((out_features, in_features), rate=rate)
		weight[~self.mask] = 0.0
		self.weight = nn.Parameter(weight)
		if bias: 
			self.bias = nn.Parameter(torch.zeros(out_features))

	def forward(self, x):
		# Use the custom autograd function to perform the masked linear operation
		return MaskedLinearFunction.apply(x, self.weight, self.bias, self.mask)
	

class MaskedConv2dFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias, mask, stride, padding, groups):
		mask = mask.to(weight.device)
		# Save the input, weight, bias, and mask tensors for backward
		ctx.save_for_backward(input, weight, bias, mask)
		# Save the stride and padding argument for backward
		ctx.stride = stride
		ctx.padding = padding
		ctx.groups = groups
		# Perform the convolution operation using the masked weight tensor
		return F.conv2d(input, weight, bias, stride=stride, padding=padding, groups=groups)

	@staticmethod
	def backward(ctx, grad_output):
		# Retrieve the saved tensors from forward
		input, weight, bias, mask = ctx.saved_tensors
		stride = ctx.stride
		padding = ctx.padding
		groups = ctx.groups
		# Compute the gradients of the input, weight, and bias tensors
		grad_input = grad_weight = grad_bias = None
		if ctx.needs_input_grad[0]:
			# grad_input = F.conv_transpose2d(grad_output, weight * mask, bias=None)
			grad_input = torch.nn.grad.conv2d_input(input.shape, weight * mask, grad_output, stride=stride, padding=padding, groups=groups)
		if ctx.needs_input_grad[1]:
			# grad_weight = F.conv_transpose2d(grad_output, input, bias=None)
			grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
			# Apply the mask to the weight gradient to make it sparse
			grad_weight = grad_weight * mask
			grad_weight = grad_weight.to_sparse()
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum((0, 2, 3))
		return grad_input, grad_weight, grad_bias, None, None, None, None


class MaskedConv2d(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
				 stride: int=1, rate: float=0, bias: bool=True, padding: int=1,
				 mask: torch.Tensor=None, groups: int=1) -> torch.Tensor:
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size=kernel_size
		self.padding=padding
		self.groups=groups
		self.mask = mask
		self.init_mask = True if mask is not None else False
		self.stride=stride
		self.bias = None 
		
		weight = torch.nn.init.kaiming_normal_(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
		if self.mask is None: 
			self.mask = utils.get_sparse_mask((out_channels, in_channels), rate=rate, kernel=kernel_size)

		weight[~self.mask] = 0.0
		self.weight = nn.Parameter(weight)

		if bias: 
			self.bias = nn.Parameter(torch.zeros(out_channels))

	def forward(self, x):
		# Use the custom autograd function to perform the masked convolution operation
		return MaskedConv2dFunction.apply(x, self.weight, self.bias, self.mask, self.stride, self.padding, self.groups)
	
	def __repr__(self):
		return f"{self.__class__.__name__}(" \
			f"{self.in_channels}, {self.out_channels}, " \
			f"kernel_size={(self.kernel_size, self.kernel_size)}), " \
			f"padding={(self.padding, self.padding)}, " \
			f"stride={(self.stride, self.stride)}, " \
			f"bias={self.bias is not None}, " \
			f"mask={self.init_mask})" 
