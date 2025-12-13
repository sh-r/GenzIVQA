# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

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
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, CLIPTextModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, DiffusionPipeline, UNet2DConditionModel, DSDPipeline
from model import DSD_Model
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utils import *
import datetime
import logging



from write_2_csv_file import write_csv_file
from scipy import stats
from antonym_prompt_learner import CustomCLIP
from flive import DataLoader
import clive

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.15.0.dev0")

import torch.utils.checkpoint as checkpoint

logger = get_logger(__name__, log_level="INFO")



def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--use_validation_split",
        action="store_true",
        help=(
            "Whether or not to use a validation split from the training dataset. Will use the validation percentage"
            " set in `args.validation_split_percentage`."
        ),
    )
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help=(
            "Whether to load the best model based on some params"
        ),
    )
    
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=0.1,
        help=(
            "The percentage of the train dataset to use as validation dataset if `args.use_validation_split` is set."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vanilla_finetuning",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--accelerator_ckpts_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the accelerator predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--unet_ckpts_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the unet weights and checkpoints will be written.",
    )

    parser.add_argument(
        "--coop_ckpts_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the coop weights and checkpoints will be written.",
    )

    

    parser.add_argument(
        "--test_res_save_path",
        type=str,
        default="results",
        help="The output directory where the quality scores for the test set will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument('--train_data', type=str, default='ComVG')
    parser.add_argument('--val_data', type=str, default='ComVG_sub')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--best_validation_epoch", type=int, default=0, required=True)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1000,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Experiment with different tokenizer"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument('--bias', action='store_false', help='Set bias to False')
    parser.add_argument('--validation', action='store_true', help='Set validation during training to True')
    parser.add_argument('--val_num', type=int, default=100, help='Number of validation used')
    parser.add_argument('--sampling_time_steps', type=int, default=30)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--calculate_srocc', action='store_true', help='Whether to calculate SROCC for the test data')



    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    # # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start


def average_score(tensor_lists):
    means = [torch.mean(tensor[0]) for inner_list in tensor_lists for tensors in inner_list for tensor in tensors]
    return sum(means) / len(means)



def rename_image_based_on_score(score, img_name, img_idx, text_folder_path):
    img_name = img_name[0].split('/')[-1]
    
    if img_idx == 0:
        img_path = os.path.join(text_folder_path, 'positive', os.path.basename(img_name))
        folder = 'positive'
    else:
        img_path = os.path.join(text_folder_path, 'negative', os.path.basename(img_name))
        folder = 'negative'
    
    if os.path.exists(img_path):
        new_img_name = f"{os.path.splitext(img_name)[0]}_score_{score}{os.path.splitext(img_name)[1]}"
        new_img_path = os.path.join(text_folder_path, folder, new_img_name)
        os.rename(img_path, new_img_path)





def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    os.makedirs(args.unet_ckpts_dir, exist_ok=True)
    os.makedirs(args.accelerator_ckpts_dir, exist_ok=True)
    os.makedirs(args.test_res_save_path, exist_ok=True)
    os.makedirs(args.coop_ckpts_dir, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('./logs', exist_ok=True)
    log_filename = f'./logs/antonym_prompt_train_on_{args.train_data}_test_on_{args.test_data}_{now}.log'
    logging.basicConfig(level=logging.INFO,
                    format=f'%(asctime)s [Process {accelerator.process_index}] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    
    
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
            

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )


    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info(f"Using {weight_dtype} for weights.")
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, 
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )
    
    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Prompt Learning Model Declaration

    coop_Model = CustomCLIP(tokenizer, text_encoder, ['Good Photo', 'Bad Photo'], 'end')

    coop_Model.to(accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = list(coop_Model.prompt_learner.parameters()) + list(lora_layers.parameters())
    
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    dir_loc = 'path_to_train_dataset'

    dataset = DataLoader(args.train_batch_size, 'FLIVE', dir_loc)
    train_dataloader, _, test_dataloader = dataset.get_data()

    # Doing test for CLIVE dataset.
    if args.train_data != args.test_data:
        test_dir_loc = 'path_to_test_dataset'
        test_dataset = clive.DataLoader(args.train_batch_size, 'CLIVE', test_dir_loc, 1162)
        test_dataloader, _, _ = test_dataset.get_data()

    length_of_train_dataset = 0
    length_of_test_dataset = 0

    for _, (_, mos_scores, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        length_of_train_dataset += len(mos_scores)

    for _, (_, mos_scores, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        length_of_test_dataset += len(mos_scores)


    print('\n' * 2)
    logger.info(f'Train dataset name: {args.train_data} and Test dataset name: {args.test_data}')
    logger.info(f'Number of training batches = {len(train_dataloader)}')
    logger.info(f'Number of training samples = {length_of_train_dataset}')
    logger.info(f'Number of test batches = {len(test_dataloader)}')
    logger.info(f'Number of test samples = {length_of_test_dataset}')
    print('\n' * 2)


    logger.info(f'Test Batch Size = {args.val_batch_size}')
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    coop_Model.prompt_learner, lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        coop_Model.prompt_learner, lora_layers, optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    #num_update_steps_per_epoch = math.ceil(len(train_latent_dataset) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("DSD-fine-tune", config=vars(args))


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {length_of_test_dataset}")
    logger.info(f"  Num Epochs trained = {args.num_train_epochs}")

    args.checkpointing_steps = num_update_steps_per_epoch

    # Global step calculation corresponding to the best Validation epoch
    best_val_global_step = (args.best_validation_epoch + 1) * len(train_dataloader)
    dirs = os.listdir(args.coop_ckpts_dir)
    best_val_weights_path = [d for d in dirs if str(best_val_global_step) in d]
    best_val_weights_path = sorted(best_val_weights_path, key=lambda x: int(x.split("-")[-1]))
    best_val_weights_path = best_val_weights_path[0]


    coop_Model.prompt_learner.load_state_dict(torch.load(os.path.join(os.path.join(args.coop_ckpts_dir, best_val_weights_path), 'coop_Model_state_dict.pth')))
    coop_Model.eval()
    logger.info(f'Loading the CoOp checkpoints from {os.path.join(args.coop_ckpts_dir, best_val_weights_path)}')
    
    unet.load_attn_procs(os.path.join(args.unet_ckpts_dir, best_val_weights_path))
    unet.eval()
    logger.info(f'Loading the UNet checkpoints from {os.path.join(args.unet_ckpts_dir, best_val_weights_path)}')

    test_loss_coop = 0.0
    quality_score = {}
    logger.info(f'***** Undergoing Testing on {args.test_data} for model trained on {args.train_data}*****')
    
    with torch.no_grad():
        pipeline_img2img = DSDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
            unet=accelerator.unwrap_model(unet),
            torch_dtype=weight_dtype
        )
        pipeline_img2img = pipeline_img2img.to(accelerator.device)
        pipeline_img2img.set_progress_bar_config(disable=True)
        coop_Model = accelerator.unwrap_model(coop_Model)

        pipeline_img2img, coop_Model, test_dataloader = accelerator.prepare(pipeline_img2img, coop_Model, test_dataloader)

        predicted_quality = []
        actual_quality = []

        num_test_samples = 0
        for batch_idx, (images, mos_scores, filenames)in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                  
            latents_batched = vae.encode(images.to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample()
            latents_batched = latents_batched * vae.config.scaling_factor
            
            scores = compute_score_genziqa(batch_idx, args, latents_batched, 
                                   pipeline_img2img, text_model = coop_Model, 
                                   fixed_prompt_embeds=None)
            
            scores = scores.contiguous()
            accelerator.wait_for_everyone()
            scores = accelerator.gather(scores)
            
            GT_MOS_Scores = mos_scores.to(accelerator.device, dtype = weight_dtype)
            
            if scores.dim() > 1:
                scores = scores.squeeze(0)
            if GT_MOS_Scores.dim() > 1:
                GT_MOS_Scores = GT_MOS_Scores.squeeze(0)

            
            loss = F.mse_loss(scores, GT_MOS_Scores)
            test_loss_coop += loss.detach().item() * len(GT_MOS_Scores)
            num_test_samples += len(GT_MOS_Scores)


            scores_list = [value.item() for value in scores]
            predicted_quality += scores_list

            mos_scores_list = [value.item() for value in GT_MOS_Scores]
            actual_quality += mos_scores_list

            for filename_idx in range(len(GT_MOS_Scores)):
                filename = filenames[filename_idx]

                if filename not in quality_score:
                    quality_score[filename] = {}

                quality_score[filename]['prediction'] = scores[filename_idx].detach().item()
                quality_score[filename]['target'] = GT_MOS_Scores[filename_idx].detach().item()

        del pipeline_img2img
        torch.cuda.empty_cache()

    test_loss_coop = test_loss_coop / num_test_samples
    test_srocc_score = (stats.spearmanr(np.array(actual_quality), np.array(predicted_quality))).statistic
    test_plcc_score = (stats.pearsonr(np.array(actual_quality), np.array(predicted_quality))).statistic
    
    logger.info(f'For the model trained on {args.train_data} and tested on {args.test_data},  Test loss: {test_loss_coop}, Test SROCC: {test_srocc_score}, Test PLCC: {test_plcc_score}')

    quality_score_csv_file_path = os.path.join(args.test_res_save_path, f'Test_on_{args.test_data}_quality_scores_for_training_on_{args.train_data}_for_bve_{args.best_validation_epoch}.csv')
    write_csv_file(quality_score, quality_score_csv_file_path)
    logger.info(f'Quality Predictions for the Test set written at the locaton: {quality_score_csv_file_path}')
    logger.info("***** End Evaluation *****")


if __name__ == "__main__":
    main()