"""
Dataloaders and dataset classes defined. 

dataset means and standard deviations computed as follows

	from torch.utils.data import ConcatDataset
	transform = transforms.Compose([transforms.ToTensor()])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
								download=True, transform=transform)

	#stack all train images together into a tensor of shape 
	#(50000, 3, 32, 32)
	x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

	#get the mean of each channel            
	mean = torch.mean(x, dim=(0,2,3)) #tensor([0.4914, 0.4822, 0.4465])
	std = torch.std(x, dim=(0,2,3)) #tensor([0.2470, 0.2435, 0.2616])  
"""


import numpy as np
import os, glob, PIL
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from helpers.lidc_dataset import LIDC_IDRI
from helpers.isic_dataset import ISIC
import medmnist 


class TargetToFloat:
	def __call__(self, target):
		return torch.tensor(target, dtype=torch.float32)
	
class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = float(std)
		self.mean = float(mean)
		
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(root: str, dataset: str, kfold: int, batch_size: int, num_points: int, 
				   train_shuffle: bool=True, run_val: bool=True, train_split: float=1, 
				   flatten_input: bool=False, input_size: int=0, test: bool=False, noise=None):

	train_sampler, test_sampler = None, None 

	if dataset.upper() == "ISIC": 
		transform = transforms.Compose([
			transforms.Normalize((0.7101, 0.5731, 0.5222), (0.1499, 0.1678, 0.1823))
		])
		if not test: 
			train_data = ISIC(root=root, split='train', input_size=input_size, transform=transform, augment=True)
			test_data = ISIC(root=root, split='val', input_size=input_size, transform=transform)
		else:
			test_data = ISIC(root=root, split='test', input_size=input_size, transform=transform)

	elif dataset.upper() == "CHESTMNIST": 
		transform_target = transforms.Compose([TargetToFloat()])
		transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4980,), (0.2471,))
			])
		if not test: 
			train_data = medmnist.ChestMNIST(root=root, split='train', download=True, size=input_size, transform=transform, target_transform=transform_target)
			test_data = medmnist.ChestMNIST(root=root, split='val', download=True, size=input_size, transform=transform, target_transform=transform_target)
		else:
			test_data = medmnist.ChestMNIST(root=root, split='test', download=True, size=input_size, transform=transform, target_transform=transform_target)
				
	elif dataset.upper() == 'LIDC' or dataset.upper() == 'LIDC_IDRI':
		if kfold > 1:
			dataset_ = LIDC_IDRI(root)
			train_data = dataset_
		else:
			dataset_ = LIDC_IDRI(root)
			dataset_size = len(dataset_)

			#split data
			indices = list(range(dataset_size))
			train_indices = indices[:int(np.floor(train_split * dataset_size))]

			train_data = dataset_
			train_sampler = SubsetRandomSampler(train_indices)
			train_shuffle = False

			if run_val: 
				test_indices = indices[len(train_indices):]
				test_sampler = SubsetRandomSampler(test_indices)
				test_data = dataset_ 
	else:
		raise Exception(f"Dataset '{dataset}' has no dataloader implemented") 

	if not test: 
		if kfold > 1: 
			return train_data, None, batch_size
		
		train_loader = DataLoader(dataset=train_data, batch_size=batch_size, 
							shuffle=train_shuffle, sampler=train_sampler)
	if run_val or test: 
		if kfold > 1 and test: 
			return train_data
		test_loader = DataLoader(dataset=test_data, batch_size=1, 
						   shuffle=False, sampler=test_sampler)
		if test:
			return test_loader
		else:
			return train_loader, test_loader, batch_size
	else:
		return train_loader, None, batch_size