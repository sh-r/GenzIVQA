import torch
import torch.nn as nn
import os
import csv
import numpy as np
import math
from tqdm import tqdm
import re
import torch.optim as optim
from scipy.stats import spearmanr
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Tuple, Union
from torch import Tensor

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        output_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(input_channels, squeeze_channels, bias=True)
        self.fc2 = torch.nn.Linear(squeeze_channels, output_channels, bias=True)
        
        self.fc3 = torch.nn.Linear(input_channels, output_channels, bias=True)
        
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        # scale = self.avgpool(input)
        scale = self.fc1(input)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        lin_input = self.fc3(input)
        return scale * lin_input

class AttentionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.k_down = nn.Linear(in_features, 64, bias=True)
        self.k_up = nn.Linear(64, out_features, bias=True)
        # self.v = nn.Linear(in_features, out_features, bias=True)
        # self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=True) 
        
        self.sigmoid = nn.Sigmoid()
    def __call__(self, x):
        x = self.norm(x)
        x = self.k_down(x)
        key = self.k_up(x)
        # val = self.v(x)
        # key = self.dropout(key)
        return key
        
class AttentionBlock_LoRA(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.k_down = nn.Linear(in_features, 64, bias=True)
        self.k_up = nn.Linear(64, out_features, bias=True)
        # self.v = nn.Linear(in_features, out_features, bias=True)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=True) 
        
        self.sigmoid = nn.Sigmoid()
        
    def __call__(self, x):
        x = self.norm(x)
        x = self.k_down(x)
        key = self.k_up(x)
        
        return key
    
class AttentionBlock_SqueezeExcitation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=True) 
        
        squeeze_channels = 64 
        self.network = SqueezeExcitation(in_features, squeeze_channels, out_features)
    def __call__(self, x):
        x = self.norm(x)
        key = self.network(x)
        return key
