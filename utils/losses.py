import datasets
import json
import argparse
import logging
import math
import os
import shutil
import time
import random
from pathlib import Path
from typing import Optional
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import re
from torch import nn
block_names = {'down_block_0_attn_0': 320,
               'down_block_0_attn_1': 320,
               'down_block_1_attn_0': 640,
               'down_block_1_attn_1': 640,
               'down_block_2_attn_0': 1280,
               'down_block_2_attn_1': 1280,
               'mid_block_attn_0': 1280,
               'up_block_1_attn_0': 1280,
               'up_block_1_attn_1': 1280,
               'up_block_1_attn_2': 1280,
               'up_block_2_attn_0': 640,
               'up_block_2_attn_1': 640,
               'up_block_2_attn_2': 640,
               'up_block_3_attn_0': 320,
               'up_block_3_attn_1': 320,
               'up_block_3_attn_2': 320
}


def head_to_batch_dim(tensor, out_dim=3):
    batch_size, seq_len, dim = tensor.shape
    head_size = dim // 64
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    
    tensor = tensor.permute(0, 2, 1, 3)

    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

    return tensor
def sanitize_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.replace(' ', '_')  
    return text[:50]


def linear_scale_factor(query_feats, filename_in_bytes, slowAttentionBlocks, fastAttentionBlocks, scale_range, linearBlocks, device):
    scaling_factor_batched = []
   
    query_feats_dict = {}
    slowFeats = []
    fastFeats = []
    for i, key in enumerate(block_names.keys()):
        temp = []
        for item in query_feats:
            temp.append(item[i])
        temp = torch.stack(temp)
        
        query_feats_dict[key] = temp

    filename = str(filename_in_bytes).split("\'")[1]
    filename_without_extn_list = filename.split('.')
        
    filename_without_extn = filename_without_extn_list[0]
    for item in filename_without_extn_list[1:]:
        if item != 'mp4':
            filename_without_extn += '.' + item

    slow_fileLoc = os.path.join('directory_containing_the_slowNet_features', filename_without_extn + '.npy') # _crf_10_ss_00_t_20.0
    fast_fileLoc = os.path.join('directory_containing_the_fastNet_features', filename_without_extn + '.npy')
    
    if not os.path.isfile(slow_fileLoc):
        return None

    slowFeats.append(torch.cat([torch.from_numpy(np.load(slow_fileLoc)).unsqueeze(0)] * 2))
    fastFeats.append(torch.cat([torch.from_numpy(np.load(fast_fileLoc)).unsqueeze(0)] * 2))

    slowFeats = torch.stack(slowFeats).to(device)
    fastFeats = torch.stack(fastFeats).to(device)

    slowAttentionBlocks.to(device)
    fastAttentionBlocks.to(device)

    similarity_scores_slow = []
    similarity_scores_fast = []

    similarity_scores_slowfast = []

    for i, (key, module) in enumerate(slowAttentionBlocks.named_children()):
        
        queryFeats = query_feats_dict[key].permute(1, 0, 2)
        queryFeats = head_to_batch_dim(queryFeats)

        keySlowFeats = module(slowFeats).squeeze(0)
        
        keySlowFeats = head_to_batch_dim(keySlowFeats)

        attentionScores_slow = (queryFeats @ keySlowFeats.transpose(-1, -2)) / (64 ** -0.5)

        similarity_scores_slow.append(attentionScores_slow.mean().unsqueeze(0) / 1)              #scale_range --> default:100, lin_block:1

    for i, (key, module) in enumerate(fastAttentionBlocks.named_children()):

        queryFeats = query_feats_dict[key].permute(1, 0, 2)
        queryFeats = head_to_batch_dim(queryFeats)

        keyFastFeats = module(fastFeats).squeeze(0)
        keyFastFeats = head_to_batch_dim(keyFastFeats)

        attentionScores_fast = (queryFeats @ keyFastFeats.transpose(-1, -2)) / (64 ** -0.5)

        similarity_scores_fast.append(attentionScores_fast.mean().unsqueeze(0) / 1)

    linear_scaling_factor = []
    for i, (key, linear_module) in enumerate(linearBlocks.named_children()):
        similarity_scores_slowfast = torch.cat((similarity_scores_slow[i], similarity_scores_fast[i]), dim=0)
        
        act_temp = linear_module(similarity_scores_slowfast)
        act_temp = nn.ReLU()(act_temp)
        linear_scaling_factor.append(act_temp)

    return torch.stack(linear_scaling_factor).squeeze(-1)

def linear_scale_factor_SF(query_feats, filename_in_bytes, slowAttentionBlocks, fastAttentionBlocks, scale_range, linearBlocks, device):

    filename = str(filename_in_bytes).split("\'")[1]
    filename_without_extn_list = filename.split('.')
        
    filename_without_extn = filename_without_extn_list[0]
    # for item in filename_without_extn_list[1:]:
    #     if item != 'mp4':
    #         filename_without_extn += '.' + item

    slow_fileLoc = os.path.join('directory_containing_the_slowNet_features', filename_without_extn + '.npy') # _crf_10_ss_00_t_20.0
    fast_fileLoc = os.path.join('directory_containing_the_fastNet_features', filename_without_extn + '.npy')
    
    if not os.path.isfile(slow_fileLoc):
        return None
    
    slowFeats = torch.from_numpy(np.load(slow_fileLoc))
    fastFeats = torch.from_numpy(np.load(fast_fileLoc))
    
    slowFeats = torch.mean(slowFeats, 0).to(device)
    fastFeats = torch.mean(fastFeats, 0).to(device)
    
    combFeats = torch.concatenate((slowFeats,fastFeats )).unsqueeze(0)
    scale_score = linearBlocks(combFeats)

    return scale_score.squeeze()
    
def scale_factor(query_feats, filename_in_bytes, slowAttentionBlocks, fastAttentionBlocks, scale_range, device):
    
    scaling_factor_batched = []
    # for filename_bytes in filenames:
    query_feats_dict = {}
    slowFeats = []
    fastFeats = []
    for i, key in enumerate(block_names.keys()):
        temp = []
        for item in query_feats:
            temp.append(item[i])
        temp = torch.stack(temp)
        
        query_feats_dict[key] = temp

    filename = str(filename_in_bytes).split("\'")[1]
    filename_without_extn_list = filename.split('.')

        
    filename_without_extn = filename_without_extn_list[0]
    
    # for item in filename_without_extn_list[1:]:
    #     if item not in ['mp4', 'webm', 'yuv']:
    #         filename_without_extn += '.' + item

    # _crf_10_ss_00_t_20.0

    slow_fileLoc = os.path.join('directory_containing_the_slowNet_features', filename_without_extn + '.npy') #_crf_10_ss_00_t_20.0
    fast_fileLoc = os.path.join('directory_containing_the_fastNet_features', filename_without_extn + '.npy')
    
    # if not os.path.exists(slow_fileLoc):
    #     return(torch.ones(16, device = device))
    if not os.path.isfile(slow_fileLoc):
        return None
    
    slowFeats.append(torch.cat([torch.from_numpy(np.load(slow_fileLoc)).unsqueeze(0)] * 2))
    fastFeats.append(torch.cat([torch.from_numpy(np.load(fast_fileLoc)).unsqueeze(0)] * 2))

    
    slowFeats = torch.stack(slowFeats).to(device)
    fastFeats = torch.stack(fastFeats).to(device)

    # print(slowFeats.shape)

    slowAttentionBlocks.to(device)
    fastAttentionBlocks.to(device)

    similarity_scores_slow = []
    similarity_scores_fast = []

    for i, (key, module) in enumerate(slowAttentionBlocks.named_children()):
        
        queryFeats = query_feats_dict[key].permute(1, 0, 2)
        queryFeats = head_to_batch_dim(queryFeats)

        keySlowFeats = module(slowFeats).squeeze(0)
        
        keySlowFeats = head_to_batch_dim(keySlowFeats)

        attentionScores_slow = (queryFeats @ keySlowFeats.transpose(-1, -2)) / (64 ** -0.5)

        # S_s = attentionScores_slow.mean() / 100
        similarity_scores_slow.append(attentionScores_slow.mean() / scale_range)
        

    for i, (key, module) in enumerate(fastAttentionBlocks.named_children()):

        queryFeats = query_feats_dict[key].permute(1, 0, 2)
        queryFeats = head_to_batch_dim(queryFeats)

        keyFastFeats = module(fastFeats).squeeze(0)
        keyFastFeats = head_to_batch_dim(keyFastFeats)

        attentionScores_fast = (queryFeats @ keyFastFeats.transpose(-1, -2)) / (64 ** -0.5)

        # S_f = attentionScores_fast.mean() / 100
        similarity_scores_fast.append(attentionScores_fast.mean() / scale_range)
        

    # similarity_scores_slow = torch.stack(similarity_scores_slow)
    # similarity_scores_fast = torch.stack(similarity_scores_fast)

    # S_s = similarity_scores_slow.mean()
    # S_f = similarity_scores_fast.mean()
    scaling_factor = []
    for i in range(len(similarity_scores_fast)):
        scaling_factor.append(torch.exp(similarity_scores_fast[i]) / torch.exp(similarity_scores_slow[i]))

    # scaling_factor = torch.exp(S_f) / torch.exp(S_s)

    return torch.stack(scaling_factor)        


def compute_score_genzvqa(args, video, video_filename, model, slowAttentionBlocks, fastAttentionBlocks, scale_range, linearBlocks=None, text_model = None, fixed_prompt_embeds=None):
    latent_datas = video
        
    batchsize = args.val_batch_size
    bsz = batchsize
    scores_per_frame = []
    query_feats_per_frame = []
    
    if fixed_prompt_embeds is None:
        prompt_embeds = text_model()
    else:
        prompt_embeds = fixed_prompt_embeds
        
    for img_idx in range(0, latent_datas.shape[0], args.val_batch_size):
        latent = latent_datas[img_idx: img_idx + args.val_batch_size]
        
        bsz = latent.shape[0]
        prompt_embeds_extended = torch.cat([prompt_embeds] * bsz, dim = 0)
        
        if len(latent.shape) == 3:
            latent = latent.unsqueeze(0)
        
        score, query_feats = model(prompt_embeds=prompt_embeds_extended, image=latent, guidance_scale=args.guidance_scale, sampling_steps=args.sampling_time_steps, layer_mask = None, use_bias = args.bias, sampled_timestep = None, level = None)
        scores_per_frame.append(score)
        query_feats_per_frame.append(query_feats)

    
    scores_per_video = (torch.stack(scores_per_frame)).mean(dim=0)

    scaling_factor = linear_scale_factor(query_feats_per_frame, video_filename, 
                                  slowAttentionBlocks, fastAttentionBlocks, scale_range, linearBlocks,
                                  latent.device)
    
    if scaling_factor is None:
        return None
    refined_scores = torch.mul(scaling_factor, scores_per_video)
    return refined_scores.mean()


def compute_score_genziqa(i, args, batch, model, text_model = None, fixed_prompt_embeds=None):
    latent_datas = batch

    batchsize = args.val_batch_size
    bsz = batchsize
    scores = []

    if fixed_prompt_embeds is None:
        prompt_embeds = text_model()
    else:
        prompt_embeds = fixed_prompt_embeds

    for img_idx in range(0, latent_datas.shape[0], args.val_batch_size):
        latent = latent_datas[img_idx: img_idx + args.val_batch_size]

        bsz = latent.shape[0]
        prompt_embeds_extended = torch.cat([prompt_embeds] * bsz, dim = 0)
 
        if len(latent.shape) == 3:
            latent = latent.unsqueeze(0)

        score, _ = model(prompt_embeds=prompt_embeds_extended, image=latent, guidance_scale=args.guidance_scale, sampling_steps=args.sampling_time_steps, layer_mask = None, use_bias = args.bias, sampled_timestep = None, level = None)
        
        scores.append(score.mean())
    
    scores = torch.stack(scores, dim=0)
    
    return scores