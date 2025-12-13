
import pandas as pd
import torch.utils.data as data
import os
import csv
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import torch
import numpy as np
import scipy.io


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CLIVEdataset(data.Dataset):

    def __init__(self, root_dir, index, transform = None):
        self.datapath = root_dir

        self.csv_file = os.path.join(self.datapath, 'Data', 'AllMOS_release.mat')
        self.image_data = os.path.join(self.datapath, 'Data', 'AllImages_release.mat')
        mos_values = scipy.io.loadmat(self.csv_file)['AllMOS_release']
        mos_values = (mos_values.reshape(-1))[7:]

        image_filenames = scipy.io.loadmat(self.image_data)['AllImages_release'][7:]
        image_filenames = list(image_filenames.reshape(-1))

        

        self.samples = image_filenames
        self.tranform = transform
        self.mos_values = mos_values
        self.indx = index

    def __getitem__(self, index):
        filename = self.samples[index].item()
        sample = pil_loader(os.path.join(self.datapath, 'Images', filename))

        if self.tranform is not None:
            sample = self.tranform(sample)

        mos = self.mos_values[index]

        return sample, mos, filename
    
    def __len__(self):
        return len(self.samples)
    


class DataLoader(object):
    def __init__(self, batch_size, dataset, path, img_idx, train_split = 1, val_split = 0.0, few_shot_examples = None):
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        
        transform = transforms.Compose([
            transforms.Resize(size = (512, 512), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])   
        ])

        self.data = CLIVEdataset(root_dir=path, index=img_idx, transform=transform)

    def get_data(self):

        dataset_size = len(self.data)
        print('dataset size is ', dataset_size)

        train_indices = list(range(dataset_size))
        val_indices = []
        overall_indices = set(range(self.data.indx))
        test_indices = list(overall_indices - set(train_indices))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, sampler=test_sampler)

        return train_loader, val_loader, test_loader



        