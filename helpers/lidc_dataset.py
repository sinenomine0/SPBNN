import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from collections import OrderedDict
import numpy as np
import os
import random
import pickle

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, root, filename='data_lidc.pickle', transform=None, softlabels=False):
        self.transform = transform
        self.softlabels = softlabels

        max_bytes = 2**31 - 1
        data = OrderedDict()
        for file in os.listdir(root):
            filename = os.fsdecode(file)
            if 'lidc' in filename:
                print("Loading file", filename)
                file_path = f"{root}/{filename}"
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data
    

    @staticmethod
    def normalize_range(image):
        image_min = np.min(image)
        image_max = np.max(image)
        range = image_max - image_min

        normalized_image = (image - image_min) / range + 1e-6

        return normalized_image

    @staticmethod
    def generate_softlabels(masks, kernel=1, sigma=1, normalize=True):
        size = masks[0].shape[0]
        size = int(size/2)

        x1, x2, y1, y2 = 256-size, 256+size, 256-size, 256+size
        dum_mat = np.zeros((512,512), dtype=np.float32)

        softlabels = []
        for m in masks: 
            dum_mat[x1:x2, y1:y2] = m
            tmp = cv2.GaussianBlur(dum_mat, (kernel,kernel), sigma)
            if normalize:
                range = (tmp.max() - tmp.min())
                if range != 0:
                    tmp = (tmp-tmp.min()) / range #normalize
            softlabels.append(tmp[x1:x2, y1:y2])
        return softlabels

    @staticmethod
    def elastic_deform(img, mask = None, patch = None, alpha=(0., 1000.), sigma=(25., 35.)):
        """code by Fabian Isensee, see MIC_DKFZ/batch_generators on github."""
        if isinstance(alpha, tuple) and isinstance(sigma, tuple):
            alpha = np.random.uniform(alpha[0], alpha[1])
            sigma = np.random.uniform(sigma[0], sigma[1])

        if patch is None:
            shape = img.shape
        else:
            shape = patch

        tmp = tuple([np.arange(i) for i in shape])
        coordinates = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        for d in range(len(shape)):
            coordinates[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        
        n_dim = len(coordinates)
        offsets = []
        for _ in range(n_dim):
            offsets.append(
                gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
        offsets = np.array(offsets)
        indices = offsets + coordinates

        # if patch is None:
            #shift to the center of the image avoid issues of image being outside the canvas
        shifted_indices = []
        for i, idx in enumerate(indices): 
            shifted_indices.append(idx-idx.min())
        # else: 
        #     shifted_indices = indices
        deform_img = np.copy(img)
        deform_img = map_coordinates(deform_img, (shifted_indices[0],shifted_indices[1]), order=3, mode='constant', cval=0.0).reshape(shape)
        
        if mask is not None:
            if not isinstance(mask, list):
                mask = [mask]
            deform_mask = []
            for i, m in enumerate(mask):
                unique_labels = np.unique(m)
                result = np.zeros(m.shape, m.dtype)
                for _, c in enumerate(unique_labels):
                    res_new = map_coordinates((m == c).astype(float), (shifted_indices[0],shifted_indices[1]), order=0, mode='constant', cval=0.0)
                    result[res_new >= 0.5] = c
                deform_mask.append(result)
                
            return deform_img, deform_mask, alpha, sigma

        else:
            return deform_img, alpha, sigma

    def apply_transform(self, image, masks, augment=False):
        # Normalize instance
        # image = self.normalize_range(image)
        width=image.shape[-1]

        # Transform to tensor
        image = torch.from_numpy(image).float()
        masks = [torch.from_numpy(m).float() for m in masks] 

        if augment:
            image = image.unsqueeze(0)
            masks = [m.unsqueeze(0) for m in masks]
            #Random crop
            if random.random() > 0.5:
                crop = transforms.RandomResizedCrop(width, scale=(0.7, 1.0), ratio=(0.8, 1.1), antialias=True)
                image = crop(image)
                masks = [crop(m) for m in masks]

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                masks = [TF.hflip(m) for m in masks]

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                masks = [TF.vflip(m) for m in masks] 

            # random rotation
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                image = TF.rotate(image, angle=angle)
                masks = [TF.rotate(m, angle=angle) for m in masks]

        return image, masks

    def __getitem__(self, index):
        image = self.images[index]

        #Randomly select one of the four labels for this image
        labels = self.labels[index]

        if self.softlabels:
            labels = self.generate_softlabels(labels, kernel=self.gauss_kernel, sigma=self.gauss_sigma, normalize=True)

        if self.transform:
            image, labels = self.apply_transform(image, labels)
        else:
            image, labels = self.apply_transform(image, labels, augment=False) 

        label = labels[random.randint(0,len(labels)-1)]
        label1 = 1-label.clone()

        if len(label.shape) == 3:
            label = torch.cat((label1, label), dim=0)
        else:
            label = torch.stack((label1, label))

        if len(image.shape) < 3:
            image = torch.unsqueeze(image,0)
        if len(label.shape) < 3:
            label = torch.unsqueeze(label,0)

        series_uid = self.series_uid[index]

        return image, label, torch.stack(labels).std(dim=0, keepdim=True),  torch.stack(labels).mean(dim=0, keepdim=True)

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)