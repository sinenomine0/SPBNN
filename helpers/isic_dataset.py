import os, glob, h5py, random
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ISIC(Dataset):
	def __init__(self, root, split='train', input_size=224, transform=None, augment=False):
		self.transform = transform
		self.augment = augment
		self.data = []
		self.labels = []
		self.ids = []

		with h5py.File( f"{root}/isic/isic_{input_size}_{split}_crop.h5", 'r') as f:
			keys = sorted(f.keys())
			for key in keys:
				self.ids.append(key)
				self.data.append(f[key]['image'][:])
				self.labels.append(f[key]['mask'][:])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.labels[index]
		img, target = torch.Tensor(img), torch.Tensor(target)
		target0 = torch.ones_like(target) - target
		target = torch.stack((target0, target))

		if self.transform is not None: 
			if img.shape[-1] == 3:
				img = img.permute(-1, 0, 1)
			img = self.transform(img)
		
		if self.augment:
			if random.random() > 0.5:
				img, target = TF.hflip(img), TF.hflip(target)
			if random.random() > 0.5:
				img, target = TF.vflip(img), TF.vflip(target)
			if random.random() > 0.5:
				angle = random.randint(-90, 90)
				img, target = TF.rotate(img, angle=angle), TF.rotate(target, angle=angle)
		return img, target

def preprocess_data(root, input_size=224, split='train'):
	split_ = {"train": "Training", "val": "Validation", "test":"Test"}
	image_dir = os.path.join(root, f'isic/ISIC2018_Task1-2_{split_[split]}_Input')
	mask_dir = os.path.join(root, f'isic/ISIC2018_Task1_{split_[split]}_GroundTruth')
	
	output_file = f"{root}/isic/isic_{input_size}_{split}_crop.h5"

	image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

	transform = transforms.Compose([
		# transforms.ToTensor(),
		transforms.Resize(size=(input_size, input_size), antialias=True)
	])
	with h5py.File(output_file, 'w') as hf:
		for img_path in tqdm((image_files), total=len(image_files)):
			img_name = os.path.basename(img_path)
			mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_segmentation.png'))

			with Image.open(img_path) as img, Image.open(mask_path) as mask:
				crop = transforms.CenterCrop(size=min(img.size))
				img, mask = crop(img), crop(mask)
				img, mask = transform(img), transform(mask)

				# Convert PIL images to numpy arrays
				img_array = np.array(img)/255
				mask_array = np.array(mask)/255

				assert img_array.shape[0:1] == mask_array.shape[0:1]

				# Add data to HDF5 file
				group = hf.create_group(img_name.split('.')[0].split('_')[1])
				group.create_dataset('image', data=img_array)
				group.create_dataset('mask', data=mask_array)

def invert_img_norm(img):
	mean = [0.7101, 0.5731, 0.5222]
	std = [0.1499, 0.1678, 0.1823]
	inv_normalize = transforms.Normalize(
		mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
		std=[1 / std[0], 1 / std[1], 1 / std[2]]
	)
	img = inv_normalize(img)
	return transforms.ToPILImage()(img)