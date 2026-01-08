# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import os
os.environ['UNSLOTH_COMPILE_DIR'] = '/tmp/unsloth_compile'
import argparse
import gc
import pickle as pkl
import pprint
import random
import shutil
from contextlib import redirect_stdout
from datetime import datetime
from time import time

import datasets.features as dsf
from datasets import Sequence
from torch.utils.data import DataLoader, distributed
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

if not hasattr(dsf, "List"):
    dsf.List = Sequence
import roboverse
import torch
import tqdm
from torch import autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import tempfile
from rv_train import models
from rv_train.configs import get_cfg_defaults
from rv_train.models.qwen.dataset import QwenSFTDataset, QwenCachedDataset
from rv_train.models.qwen.dataset_preprocess import preprocess_qwen_dataset
from rv_train.models.qwen.model import QwenActor
from rv_train.utils import train_utils as utils
from accelerate import Accelerator
DEVICE = ""

START_TIME = time()

import os
import pickle as pkl

from transformers import TrainerCallback

os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1"
"""
Training script usage:

# Resume training on single GPU with a LoRA adapter
python -m rv_train.train \
  --exp-config ./configs/vla0.yaml \
  --resume \
  --model-path /workspace/runs/vla0/checkpoint-1600 \
  --devices 0

# Train from scratch
python -m rv_train.train --exp-config ./configs/vla0.yaml

# Multi GPU
accelerate launch --num_processes=4   -m rv_train.train   --exp-config ./configs/vla0.yaml   --resume   --model-path /workspace/runs/vla0/checkpoint-1600
Notes:
- The --resume flag allows loading either a full model or a LoRA adapter.
- Dataset statistics are saved automatically:
    dataset_stats_path = os.path.join(log_dir, "dataset_stats.pkl")
    with open(dataset_stats_path, "wb") as f:
        pkl.dump(train_dataset.stats, f)
    print(f"Dataset stats saved to {dataset_stats_path}")
- The final model is saved via trainer.save_model().
- Multi-GPU training is supported via --devices argument and mp.spawn.
"""


class DatasetStatsCallback(TrainerCallback):
    def __init__(self, model, log_dir):
        self.model = model
        self.log_dir = log_dir

    def on_save(self, args, state, control, **kwargs):
        try:
            accelerator = Accelerator()
            if not accelerator.is_main_process:
                return
        except:
            pass
            
        stats_path = os.path.join(self.log_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pkl.dump(self.model.original_dataset_stats, f)
        print(f"Dataset stats saved to {stats_path}")

def get_pretrained_model(
    model_path: str,
    device=0,
):
    device = f"cuda:{device}" if isinstance(device, int) else device

    model_folder = os.path.dirname(
        model_path
    )  # .../full_model/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit
    version_folder = os.path.basename(
        model_path
    )  # train_steps10_lora16_Batch8_GradAccu2_Horizon8_v0
    parent_model_name = os.path.basename(
        model_folder
    )  # Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit

    cfg_path = os.path.join(model_folder, f"{parent_model_name}_config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    cfg = get_cfg(cfg_path, cfg_opts="")

    model = get_model(
        cfg,
        calculate_dataset_stats=False,
        for_training=False,
    ).to(device)
    stats_path = os.path.join(os.path.dirname(model_path), "dataset_stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            dataset_stats = pkl.load(f)
            model.set_dataset_stats(dataset_stats)
    model.eval()

    print("Pretrained model ready for eval & inference")

    return model, cfg


def get_cfg(cfg_path="", cfg_opts=""):
    """
    Loads config with optional overrides.

    Automatically switches to full_model config if cfg_path points
    to a model folder under runs/full_model.

    :param cfg_path: Path to config.yaml or model folder
    :param cfg_opts: Overrides from command line
    """
    cfg = get_cfg_defaults()

    if cfg_path and os.path.isdir(cfg_path) and "full_model" in cfg_path:
        model_name = os.path.basename(cfg_path)
        cfg_path_candidate = os.path.join(cfg_path, f"{model_name}_config.yaml")
        if os.path.exists(cfg_path_candidate):
            cfg_path = cfg_path_candidate
        else:
            raise FileNotFoundError(
                f"Config for inference not found at {cfg_path_candidate}"
            )

    if cfg_path != "":
        cfg.merge_from_file(cfg_path)

    if cfg_opts != "":
        cfg.merge_from_list(cfg_opts.split(" "))
        cfg.EXP.EXP_ID += f"_{utils.short_name(cfg_opts)}"

    cfg.freeze()
    print(cfg)
    return cfg


def get_inp(cfg, data_batch):
    """
    Constructs the input for the model using the batched data.
    :param cfg: config object
    :param data_batch: contains the batched data provided by the dataloader
    """

    inp = data_batch
    return inp

def get_model(cfg, calculate_dataset_stats: bool = True, for_training: bool = True):

    model = models.QwenActor(**cfg.MODEL.UNSLOTH, for_training=for_training)

    if calculate_dataset_stats and for_training:
        accelerator = Accelerator()
        stats = None

        if accelerator.is_main_process:
            print("Computing dataset stats on main process...")
            temp_dataset = get_dataloader(split="train", cfg=cfg, get_dataset=True)
            stats = temp_dataset.stats
            del temp_dataset
            temp_path = os.path.join(tempfile.gettempdir(), "dataset_stats.pkl")
            with open(temp_path, "wb") as f:
                pkl.dump(stats, f)

        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            temp_path = os.path.join(tempfile.gettempdir(), "dataset_stats.pkl")
            with open(temp_path, "rb") as f:
                stats = pkl.load(f)

        model.set_dataset_stats(stats)
        accelerator.wait_for_everyone()
    return model


def default_batch_proc(data_batch, device):
    for x in data_batch:
        if isinstance(data_batch[x], dict):
            for y in data_batch[x]:
                data_batch[x][y] = data_batch[x][y].to(device).float()
        else:
            if isinstance(data_batch[x], torch.Tensor):
                data_batch[x] = data_batch[x].to(device).float()
            else:
                data_batch[x] = data_batch[x]
    return data_batch


def get_dataloader(split, cfg, get_dataset=False):
    """
    Returns dataloader based on the config and split
    :param get_dataset: whether to return the dataset or the dataloader
    """
    num_workers = cfg.DATALOADER.num_workers
    dataset_args = {"split": split}
    if cfg.EXP.DATASET == "roboverse":
        print("WARNING: split is ignored for roboverse dataset.")
        dataset_args = dict(**cfg.DATALOADER.ROBOVERSE)
        dataset = roboverse.get_unified_dataset(**dataset_args)
    else:
        raise NotImplementedError
    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = default_batch_proc
    if get_dataset:
        return dataset
    else:
        return DataLoader(
            dataset,
            num_workers=num_workers,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,  # ← NEU: Prefetch 2 batches
        )

def check_grad(model, loss):
    bad_grad = False
    if loss.ne(loss).any():
        bad_grad = True
        print("WARNING: nan in the loss")
    else:
        for x in model.parameters():
            if x.grad is not None:
                if x.grad.ne(x.grad).any():
                    print("WARNING: nan in a gradient")
                    bad_grad = True
                    break
                if ((x.grad == float("inf")) | (x.grad == float("-inf"))).any():
                    print("WARNING: inf in a gradient")
                    bad_grad = True
                    break
    return bad_grad


def print_model_stats(model):
    """Print model statistics including parameter counts."""
    model_module = model.module if isinstance(model, DDP) else model
    total_params = sum(p.numel() for p in model_module.parameters())
    trainable_params = sum(
        p.numel() for p in model_module.parameters() if p.requires_grad
    )
    non_trainable_params = total_params - trainable_params

    print("=" * 50)
    print("Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 50)


def get_log_dir(cfg, logdir_with_time=False):
    if logdir_with_time:
        log_dir = (
            f"./runs/{cfg.EXP.EXP_ID}/{str(datetime.now())[:-7].replace(' ', '-')}"
        )
    else:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    return log_dir


def entry_train(
    cfg,
    logdir_with_time=False,
    resume=False,
    model_path="",
):
    accelerator = Accelerator()
    device = "cuda" if torch.cuda.is_available() else "cpu"


    log_dir = get_log_dir(cfg, logdir_with_time)
    os.makedirs(log_dir, exist_ok=True)


    model = get_model(cfg, calculate_dataset_stats=not resume, for_training=True)

    loader_train = get_dataloader(split="train", cfg=cfg, get_dataset=True)
    precompute_dataset = True
    if precompute_dataset: 
        cache_file = "cache_checkpoints/cache_part_00000.pt
        train_dataset = models.QwenCachedDataset(
            cache_file=cache_file,
            preprocess_fn=lambda: models.preprocess_qwen_dataset(loader_train, model, max_workers=16)
        )
    else:
        train_dataset = models.QwenSFTDataset(loader_train, model)
    
    if resume and model_path:

        run_dir = os.path.dirname(model_path)          # /home/felix/vla01/runs/vla0/checkpoint-1600 -> checkpoint-1600
        stats_path = os.path.join(run_dir, "dataset_stats.pkl")
    
        print("model_path :", model_path)
        print("stats_path :", stats_path)
        print("exists    :", os.path.exists(stats_path))
    
        if not os.path.exists(stats_path):
            raise RuntimeError(f"dataset_stats.pkl NOT FOUND!\nExpected at: {stats_path}")
    
        with open(stats_path, "rb") as f:
            model.set_dataset_stats(pkl.load(f))


    train_dataset.model = model

    # -------------------------
    # Trainer
    # -------------------------
    stats_callback = DatasetStatsCallback(model, log_dir)

    trainer = SFTTrainer(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=train_dataset,
        data_collator=models.MaskedVisionDataCollator(
            model.model,
            model.tokenizer,
            train_on_responses_only=True,
            instruction_part="user",
            response_part="assistant",
            completion_only_loss=True,
        ),
        args=SFTConfig(
            per_device_train_batch_size=cfg.TRAIN.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.TRAIN.gradient_accumulation_steps,
            warmup_steps=cfg.TRAIN.warmup_steps,
            max_steps=cfg.TRAIN.max_steps,
            learning_rate=cfg.TRAIN.learning_rate,
            logging_steps=cfg.TRAIN.logging_steps,
            optim=cfg.TRAIN.optim,
            weight_decay=cfg.TRAIN.weight_decay,
            lr_scheduler_type=cfg.TRAIN.lr_scheduler_type,
            seed=cfg.TRAIN.seed,
            output_dir=log_dir,
            report_to=cfg.TRAIN.report_to,
            remove_unused_columns=cfg.TRAIN.remove_unused_columns,
            dataset_text_field=cfg.TRAIN.dataset_text_field,
            dataset_kwargs=dict(cfg.TRAIN.dataset_kwargs),
            max_length=cfg.TRAIN.max_length,
            save_steps=cfg.TRAIN.save_steps,
            save_total_limit=cfg.TRAIN.save_total_limit,
            ddp_find_unused_parameters=False,
            gradient_checkpointing="selective",
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )
    trainer.add_callback(stats_callback)
    
    trainer.train(resume_from_checkpoint=model_path if resume else None)
    trainer.save_model()
    
    if accelerator.is_main_process:
        stats_path = os.path.join(log_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pkl.dump(train_dataset.stats, f)
        print(f"Saved dataset stats to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=str, default="train")
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("--exp-cfg-opts", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--logdir-with-time", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=False)

    cmd_args = parser.parse_args()

    # from rv_train.configs import get_cfg

    _cfg = get_cfg(cmd_args.exp_config, cmd_args.exp_cfg_opts)

    entry_train(
        cfg=_cfg,
        logdir_with_time=cmd_args.logdir_with_time,
        resume=cmd_args.resume,
        model_path=cmd_args.model_path,
    )


