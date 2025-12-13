####################################  Load official FLIVE train-val-test split  ####################################


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

class FLIVEdataset(data.Dataset):

    def __init__(self, root_dir, index, transform = None):
        self.datapath = root_dir

        self.csv_file = os.path.join(self.datapath, 'labels_image.csv')
        mos_values = []
        sample = []
        
        with open(self.csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample.append(row['name'])
                mos_values.append(float(row['mos']))

        

        self.samples = sample
        self.tranform = transform
        self.mos_values = mos_values
        self.indx = index

    def __getitem__(self, index):
        filename = self.samples[index]

        sample = pil_loader(os.path.join(self.datapath, filename))

        if self.tranform is not None:
            sample = self.tranform(sample)

        mos = float(self.mos_values[index])

        return sample, mos, filename
    
    def __len__(self):
        return len(self.samples)
    


class DataLoader(object):
    def __init__(self, batch_size, dataset, path, img_idx = 39810, train_split = 1, val_split = 0.0, few_shot_examples = None):
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.datapath = path


        transform = transforms.Compose([
            transforms.Resize(size = (512, 512), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])   
        ])
        self.data = FLIVEdataset(root_dir=self.datapath, index=img_idx, transform=transform)

    def get_data(self):

        dataset_size = len(self.data)
        print('dataset size is ', dataset_size)

        train_val_split_path = os.path.join(self.datapath, 'labels=640_padded.csv')
        data1 = pd.read_csv(train_val_split_path)
        data2 = pd.read_csv(self.data.csv_file)

        train_info = data1['is_valid'] == False
        train_filenames = data1[train_info]['name_image'].tolist()
        train_cdn = data2['name'].isin(train_filenames)
        train_indices = data2[train_cdn].index.tolist()

        val_info = data1['is_valid'] == True
        val_filenames = data1[val_info]['name_image'].tolist()
        val_cdn = data2['name'].isin(val_filenames)
        val_indices = data2[val_cdn].index.tolist()

        
        overall_indices = set(range(len(data2['name'])))
        test_indices = list(overall_indices - set(val_indices + train_indices))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)


        train_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, sampler=test_sampler)
        
        
        return train_loader, val_loader, test_loader