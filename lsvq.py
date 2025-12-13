# ----------------------------Load the official LSVQ train set----------------------------

import skvideo
import skvideo.io
import os
import csv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from create_hdf5_file import create_hdf5_file
from torchvision.transforms.functional import InterpolationMode
import random

from diffusers import AutoencoderKL


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder="vae")
vae.to(device)

video_directory = 'path_to_the_dataset'
csv_info = os.path.join(video_directory, 'train_labels.txt')

samples = []
mos_scores = []

transform = transforms.Compose([
                transforms.Resize(size = (512, 512), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])   
            ])


with open (csv_info, 'r') as f:
    for line in f:
        info = line.strip().split(', ')
        filename = info[0]
        mos = info[-1]
        mos_scores.append(float(mos))
        samples.append((filename, float(mos)))

video_frames = {}
frames_per_video = 12

random.shuffle(samples)

datas_cache = []
mos_scores_cache = []
filenames_cache = []

no_videos = []

count = 0
for i, sample in tqdm(enumerate(samples), total=len(samples)):
    filename, mos = sample
    file_loc = video_directory + filename

    if not os.path.isfile(file_loc):
        no_videos.append(file_loc)
        continue

    get_video = skvideo.io.vread(file_loc)           
    videometadata = skvideo.io.ffprobe(file_loc)
    desired_frame_indices = list(np.linspace(0, len(get_video) - 1, frames_per_video).astype(int))

    extracted_frames = []
    
    for j in range(0, frames_per_video):
        frame = Image.fromarray(get_video[desired_frame_indices[j]])
        extracted_frames.append(transform(frame))
        
    extracted_frames = torch.stack(extracted_frames)
    count += 1

    with torch.no_grad():
        x0 = vae.encode(extracted_frames.to(dtype=torch.float32, device=device)).latent_dist.sample()
        x0 = x0 * vae.config.scaling_factor
        del extracted_frames

    
    datas_cache.append(x0.cpu())
    del x0
    mos_scores_cache.append(float(mos))
    filenames_cache.append(filename)

    if count % 16 == 0 or (i + 1) == len(samples):
        batch_idx = (count - 1) // 16
        video_frames[batch_idx] = {}
        video_frames[batch_idx]['datas'] = torch.stack(datas_cache)
        video_frames[batch_idx]['mos_scores'] = torch.tensor(mos_scores_cache).to(dtype = torch.float32).numpy()
        video_frames[batch_idx]['filenames'] = np.array(filenames_cache, dtype = 'S')
        
        datas_cache = []
        mos_scores_cache = []
        filenames_cache = []
        
        print(f'Batch {batch_idx} contains {len(video_frames[batch_idx]["mos_scores"])}, {video_frames[batch_idx]["datas"].shape}')
        
print(f'There are {count} data samples')
print(f'There are {len(no_videos)} scores whose video is not there')
create_hdf5_file('lsvq_train_latent.h5', video_frames)
    
