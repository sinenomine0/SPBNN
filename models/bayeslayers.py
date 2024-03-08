"""
Reparameterization layers from:
source: https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/layers/variational_layers/

modified for partial bayesian application

"""
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from torch.nn import Parameter
from itertools import repeat
import models.maskedtensor as mt
import collections, math
import utils

def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p, mask=None):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(sigma_q) + \
              (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
        
        #for partial bayesian, compute the kl for only bayesian nodes, 
        #deterministic nodes will have a kl of 0
        if mask is not None:
            #check when all nodes are deterministic
            if mask.sum()/mask.numel() == 1:
                return torch.tensor(0.).to(mask.device)
            return kl[~mask].mean()

        return kl.mean()


class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 prior_mean: float=0.,
                 prior_variance: float=1.,
                 posterior_mu_init: float=0.,
                 posterior_rho_init: float=-3.,
                 bias: bool=True, 
                 sparse: bool=False,
                 mask: list=[],
                 init_param: list=[],
                 fit_variance_only: bool=False,
                 test: bool=False):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
            sparse: bool -> if set to True, then a the weights and their associated rho will be set to zero and sparsified, 
                            either at random if no mask is provided, or according to the provided mask,
            mask: list -> list of tensors of the same shape as the weight and bias (if applicable) for sparsifying the gradients or weights,
            init_param: list -> list of the same length as the mask, if provided, it indicates that a partial Bayes layer is to be declared, and the 
                                weight means are initialized with the values from the init_param,
            fit_variance_only: bool -> when init_param is provided, this parameter is set to True, then only the active rho nodes would be fit, the 
                                weight means are not refitted.
        """
        super(LinearReparameterization, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        self.sparse = sparse['sparse']
        self.fit_variance_only = fit_variance_only
        self.test = test 
        if self.sparse:
            sparse_rate = sparse['rate']
            if not mask: 
                mask = [utils.get_sparse_mask((out_features, in_features), rate=sparse_rate)]
        if mask:
            mask = [m.to(torch.bool) for m in mask]

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features)) 
        if fit_variance_only: 
            if not init_param or not mask: 
                raise Exception("Attribute fit_variance_only set to True, but init_param or mask (or both) are empty")
            self.mu_weight.requires_grad = False      
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))

        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features), persistent=False)
        if mask:
            self.register_buffer('weight_mask', mask[0], persistent=True)
        else:
            if test:
                #to be able to load state_dict if the model is partial bayes
                self.register_buffer('weight_mask', 
                                    torch.zeros(self.mu_weight.shape).to(torch.bool), 
                                    persistent=True)
            else:
                self.register_buffer('weight_mask', None, persistent=True)
        if init_param:
            self.register_buffer('init_weight_mu', init_param[0], persistent=False)
        else: 
            self.register_buffer('init_weight_mu', None, persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            if self.fit_variance_only: 
                if not init_param or not mask: 
                    raise Exception("Attribute fit_variance_only set to True, but init_param or mask (or both) are empty")
                self.mu_bias.requires_grad = False

            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_features), persistent=False)
            if mask and len(mask) > 1:
                self.register_buffer('bias_mask', mask[1], persistent=True)
            else:
                if test:
                    #to be able to load state_dict if model is partial bayes
                    self.register_buffer('bias_mask', 
                                        torch.zeros(self.mu_bias.shape).to(torch.bool), 
                                        persistent=True)
                else:
                    self.register_buffer('bias_mask', None, persistent=True)
            if init_param:
                self.register_buffer('init_bias_mu', init_param[1], persistent=False)
            else: 
                self.register_buffer('init_bias_mu', None, persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('init_bias_mu', None, persistent=False)
            self.register_buffer('bias_mask', None, persistent=False)
        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)
        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=1)
        if self.init_weight_mu is not None or (self.test and self.weight_mask is not None):
            self.prior_weight_sigma[self.weight_mask] = math.log1p(math.exp(1e-10)) #1e-10 is used for numerical stability for log computation
            if not self.test:
                self.mu_weight.data = self.init_weight_mu
            self.rho_weight.data[self.weight_mask] = 1e-10
        if self.sparse:
            self.mu_weight.data[self.weight_mask] = 0.0
            self.rho_weight.data[self.weight_mask] = 1e-10
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],std=1)
            if self.init_bias_mu is not None or (self.test and self.bias_mask is not None):
                if not self.test:
                    self.mu_bias.data = self.init_bias_mu
                self.rho_bias.data[self.bias_mask] = 1e-10

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl = kl +  self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, input, return_kl=True):
        if self.test: 
            self.weight_mask = self.weight_mask.to(torch.bool) if self.weight_mask is not None else None 
            self.bias_mask = self.bias_mask.to(torch.bool) if self.bias_mask is not None else None 

        if self.dnn_to_bnn_flag:
            return_kl = False

        eps_weight = self.eps_weight.data.normal_()
        sigma_weight = torch.log1p(torch.exp(self.rho_weight)) 
        sigma_weight_sample = sigma_weight * eps_weight
        if self.init_weight_mu is not None or (self.test and self.weight_mask is not None):
            sigma_weight_sample = ~self.weight_mask * sigma_weight_sample
        weight = self.mu_weight + sigma_weight_sample
        
        bias = sigma_bias_sample = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))  
            sigma_bias_sample = sigma_bias * eps_bias 
            if self.init_bias_mu is not None or (self.test and self.bias_mask is not None):
                sigma_bias_sample = ~self.bias_mask * sigma_bias_sample
            bias = self.mu_bias + sigma_bias_sample

        if self.init_weight_mu is not None or (self.test and self.weight_mask is not None):
            bias_mask = ~self.bias_mask if self.bias else self.bias #to cover the usecase where bias is false to avoid error with ~ operator
            out = mt.MaskedLinearReparameterizationFunction.apply(input, self.mu_weight, sigma_weight_sample, ~self.weight_mask, 
                                                                            self.mu_bias, sigma_bias_sample, bias_mask)
        elif self.sparse: 
            out = mt.MaskedLinearFunction.apply(input, weight, bias, self.weight_mask)
        else:
            out = F.linear(input, weight, bias)

        if return_kl:
            kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu, 
                             self.prior_weight_sigma, self.weight_mask)
            if self.bias:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, 
                                      self.prior_bias_sigma, self.bias_mask)
            return out, kl
        return out
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" \
            f"{self.in_features}, {self.out_features}, " \
            f"bias={self.bias is not None}, "\
            f"sparse={str(self.sparse)}, " \
            f"initialized={str(True) if self.init_weight_mu is not None else str(False)})"


class Conv2dReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int=1,
                 padding: int=0,
                 dilation: int=1,
                 groups: int=1,
                 prior_mean: float=0.,
                 prior_variance: float=1.,
                 posterior_mu_init: float=0.,
                 posterior_rho_init: float=-3.,
                 bias: bool=True,
                 sparse: bool=False,
                 mask: list=[],
                 init_param: list=[],
                 fit_variance_only: bool=False,
                 test: bool=False):
        """
        Implements Conv2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
            sparse: bool -> if set to True, gradient w.r.t. weight matrix will be a sparse tensor. Default: False,
            mask: torch.Tensor -> a mask of shape [out_channels, in_channels // groups, kernel_size[0], kernel_size[1]] to be applied on the gradient matrix. Default: None,
        """

        super(Conv2dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias
        self.sparse = sparse['sparse']
        self.fit_variance_only = fit_variance_only
        self.test = test
        if self.sparse:
            sparse_rate = sparse['rate']
            if not mask: 
                mask = [utils.get_sparse_mask((out_channels, in_channels), rate=sparse_rate, kernel=kernel_size)]
        if not isinstance(mask, list):
            mask = [mask]     
        if not isinstance(init_param, list):
            init_param = [init_param]   
        if mask:
            mask = [m.to(torch.bool) for m in mask]
        
        kernel_size = get_kernel_size(kernel_size, 2)

        self.mu_weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if fit_variance_only: 
            if not init_param or not mask: 
                raise Exception("Attribute fit_variance_only set to True, but init_param or mask (or both) are empty")
            self.mu_weight.requires_grad = False  

        self.rho_weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        self.register_buffer(
            'eps_weight',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]),
            persistent=False)
        if mask:
            self.register_buffer('weight_mask', mask[0], persistent=True)
        else:
            if test:
                #to be able to load state_dict for partial bayes
                self.register_buffer('weight_mask', 
                                     torch.zeros(self.mu_weight.shape).to(torch.bool), 
                                     persistent=True)
            else:
                self.register_buffer('weight_mask', None, persistent=True)
        if init_param:
            self.register_buffer('init_weight_mu', init_param[0], persistent=False)
        else:
            self.register_buffer('init_weight_mu', None, persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            if self.fit_variance_only: 
                if not init_param or not mask: 
                    raise Exception("Attribute fit_variance_only set to True, but init_param or mask (or both) are empty")
                self.mu_bias.requires_grad = False
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_channels), persistent=False)
            if mask and len(mask) > 1:
                self.register_buffer('bias_mask', mask[1], persistent=True)
            else:
                if test:
                    #to be able to load state_dict for partial bayes
                    self.register_buffer('bias_mask', 
                                        torch.zeros(self.mu_bias.shape).to(torch.bool), 
                                        persistent=True)
                else:
                    self.register_buffer('bias_mask', None, persistent=True)
            if init_param:
                self.register_buffer('init_bias_mu', init_param[1], persistent=False)
            else:
                self.register_buffer('init_bias_mu', None, persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('init_bias_mu', None, persistent=False)
            self.register_buffer('bias_mask', None, persistent=False)
        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)
        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=1)
        if self.init_weight_mu is not None or (self.test and self.weight_mask is not None):
            self.prior_weight_sigma[self.weight_mask] = math.log1p(math.exp(1e-10))
            if not self.test:
                self.mu_weight.data = self.init_weight_mu
            self.rho_weight.data[self.weight_mask] = 1e-10
        if self.sparse: 
            self.mu_weight.data[self.weight_mask] = 0.0
            self.rho_weight.data[self.weight_mask] = 1e-10
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=1)
            if self.init_bias_mu is not None or (self.test and self.bias_mask is not None):
                if not self.test:
                    self.mu_bias.data = self.init_bias_mu
                self.rho_bias.data[self.bias_mask] = 1e-10

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)

        return kl

    def forward(self, input, return_kl=True):
        if self.test: 
            self.weight_mask = self.weight_mask.to(torch.bool) if self.weight_mask is not None else None 
            self.bias_mask = self.bias_mask.to(torch.bool) if self.bias_mask is not None else None 

        if self.dnn_to_bnn_flag:
            return_kl = False

        eps_weight = self.eps_weight.data.normal_()
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        sigma_weight_sample = sigma_weight * eps_weight
        if self.init_weight_mu is not None or (self.test and self.weight_mask is not None):
            sigma_weight_sample = ~self.weight_mask * sigma_weight_sample
        if self.sparse:
            sigma_weight_sample = ~self.weight_mask * sigma_weight_sample
        weight = self.mu_weight + sigma_weight_sample

        bias =  sigma_bias_sample = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            sigma_bias_sample = sigma_bias * eps_bias
            if self.init_bias_mu is not None or (self.test and self.bias_mask is not None):
                sigma_bias_sample = ~self.bias_mask * sigma_bias_sample
            bias = self.mu_bias + sigma_bias_sample

        if self.init_weight_mu is not None or (self.test and self.weight_mask is not None):
            bias_mask = ~self.bias_mask if self.bias else self.bias #to cover the usecase where bias is false to avoid error with ~ operator
            out = mt.MaskedConv2dReparameterizationFunction.apply(input, self.mu_weight, sigma_weight_sample, ~self.weight_mask, 
                                                                            self.mu_bias, sigma_bias_sample, bias_mask, 
                                                                            self.stride, self.padding, self.groups)
        elif self.sparse: 
            out = mt.MaskedConv2dFunction.apply(input, weight, bias, ~self.weight_mask, self.stride, self.padding, self.groups)
        else: 
            out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        if return_kl:
            kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu, 
                             self.prior_weight_sigma, self.weight_mask)
            if self.bias:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, 
                                      self.prior_bias_sigma, self.bias_mask)
            return out, kl    
        return out
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" \
            f"{self.in_channels}, {self.out_channels}, " \
            f"kernel_size={(self.kernel_size, self.kernel_size)}), " \
            f"padding={(self.padding, self.padding)}, " \
            f"stride={(self.stride, self.stride)}, " \
            f"bias={self.bias is not None}), " \
            f"sparse={str(self.sparse)}), " \
            f"initialized={str(True) if self.init_weight_mu is not None else str(False)})"
