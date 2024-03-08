import models.maskedtensor as mt
import torch.nn as nn
import torch.nn.functional as F
import utils
import math

def get_layer(input: int, output: int, bias: bool=False, kernel: int=1, stride: int=1,
              sparse: bool=False, layer_type: str="linear", sparse_params: dict={},
              padding: int=0):
    if sparse: 
        if layer_type == "linear":
            return mt.MaskedLinear(in_features=input, out_features=output, 
                               bias=bias, rate=sparse_params["rate"])
        elif layer_type == "conv":            
            return mt.MaskedConv2d(in_channels=input, out_channels=output, 
                                          kernel_size=kernel, stride=stride, padding=padding,
                                          bias=bias, rate=sparse_params["rate"]) 
    else:
        if layer_type == "linear":
            return nn.Linear(input, output, bias=bias)
        elif layer_type == "conv":
            return nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel, 
                             stride=stride, bias=bias, padding=padding)

class DeterministicModel(nn.Module):
    def __init__(self, input_size:int, feature_size:int, output_size=1, add_bias=True,
                 kernel_size:int=0, stride:int=1, pooling_size: int=0, nonlinearity: str='relu',  
                 batch_norm: bool=False, init_w: str='kaiming_n', sparse: dict={}, 
                 task: str="regression", layer_type: str="linear", padding: int=0):
        super().__init__() 
        self.layer_type = layer_type
        self.task = task
        nonlinearity = utils.get_nonlinearity(nonlinearity=nonlinearity)

        layers = []
        track_size = input_size
        if not isinstance(feature_size, list):
            feature_size = [input_size, feature_size]
        for i, s in enumerate(feature_size):
            if layer_type=="linear":
                if i==len(feature_size)-1:
                    layers.append(get_layer(input=s, output=output_size, bias=add_bias, 
                                            layer_type=layer_type, kernel=kernel_size, stride=stride,
                                            sparse=sparse['sparse'], sparse_params=sparse))
                else:
                    layers.append(get_layer(input=s, output=feature_size[i+1], bias=add_bias,
                                            layer_type=layer_type, kernel=kernel_size, stride=stride,
                                            sparse=sparse['sparse'], sparse_params=sparse))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(feature_size[i+1])) 
                    layers.append(nonlinearity)
            if layer_type == "conv":
                if i == len(feature_size)-1:
                    #Dense Block
                    layers.append(nn.Flatten())
                    layers.append(get_layer(input=track_size**2*feature_size[-1], output=feature_size[-1], bias=add_bias,
                                        layer_type='linear', sparse=sparse['sparse'], sparse_params=sparse))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(feature_size[-1])) 
                    layers.append(nonlinearity) 
                    layers.append(get_layer(input=feature_size[-1], output=output_size, bias=add_bias,
                            layer_type='linear', sparse=sparse['sparse'], sparse_params=sparse))
                else:
                    layers.append(get_layer(input=s, output=feature_size[i+1], bias=add_bias, sparse=sparse["sparse"], 
                                            sparse_params=sparse, layer_type=layer_type, kernel=kernel_size, 
                                            stride=stride, padding=padding))
                    if batch_norm:
                        layers.append(nn.BatchNorm2d(feature_size[i+1]))
                    layers.append(nonlinearity)
                    layers.append(get_layer(input=feature_size[i+1], output=feature_size[i+1], bias=add_bias, 
                                            sparse=sparse["sparse"], sparse_params=sparse, layer_type=layer_type, 
                                            kernel=kernel_size, stride=stride, padding=padding))    
                    if batch_norm:
                        layers.append(nn.BatchNorm2d(feature_size[i+1]))
                    layers.append(nonlinearity)
                    # track_size = math.floor(((track_size - (kernel_size - 1) -1)/stride) + 1)
                    # track_size = math.floor(((track_size - (kernel_size - 1) -1)/stride) + 1)
                    track_size = math.floor(((track_size + 2*padding - (kernel_size - 1) -1)/stride) + 1)
                    track_size = math.floor(((track_size + 2*padding - (kernel_size - 1) -1)/stride) + 1)
                    if pooling_size != 0: 
                        #Output size = [(Input size - Kernel size + 2 * Padding) / Stride] + 1
                        layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=1))
                        track_size = math.floor(((track_size - (pooling_size - 1) -1)/stride) + 1)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, nll_loss=False):
        x = self.layers(x)
        if self.task == "classification" and nll_loss:
            x = F.log_softmax(x, dim=-1)
        return x
        
